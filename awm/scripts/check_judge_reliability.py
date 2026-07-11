"""Replay AWM rollouts and inspect or measure LLM-judge reasoning.

Pipeline:
  1. collect: replay the first rollout for each unique (scenario, task_idx) into
     fresh AWM envs, run code/sql verifiers once, and save frozen SQL-verifier
     payloads including the replayed judge's reasoning. Pass --all-rollouts to
     collect every sampled trajectory instead.
  2. judge: rerun only the LLM judge N times per frozen payload and model.
  3. metrics: compute pairwise agreement, Fleiss-style kappa, and reward flips.

Run from the repo root with the AWM env server already running.

Usage:
  # 1. Replay 200 unique tasks and freeze verifier outputs.
  uv run python awm/scripts/check_judge_reliability.py collect \
    --rollouts /path/to/rollouts.jsonl \
    --output /path/to/judge_payloads.jsonl \
    --env-url http://localhost:8899 \
    --limit 200

  # 2. Rerun only the LLM judge over frozen SQL-verifier outputs.
  uv run python awm/scripts/check_judge_reliability.py judge \
    --payloads /path/to/judge_payloads.jsonl \
    --output /path/to/judge_votes.jsonl \
    --models openai/gpt-5.1 deepseek/deepseek-v4-flash \
    --votes 5

  # 3. Compute agreement, kappa, and reward-flip metrics.
  uv run python awm/scripts/check_judge_reliability.py metrics \
    --votes /path/to/judge_votes.jsonl \
    --metrics-json /path/to/judge_reliability_metrics.json \
    --summary-md /path/to/judge_reliability_summary.md
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict
import copy
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import sys
import time
from typing import Any
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
OPENENV_ROOT = (ROOT / "../OpenEnv").resolve()
for candidate in (OPENENV_ROOT / "src", OPENENV_ROOT / "envs"):
    if candidate.exists():
        sys.path.insert(0, str(candidate))

from agent_world_model_env import AWMEnv  # noqa: E402
from agent_world_model_env.server.data_loader import AWMDataLoader  # noqa: E402
from agent_world_model_env.server.verifier import run_llm_judge  # noqa: E402
from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction  # noqa: E402


VALID_LABELS = ("complete", "incomplete", "server_error", "agent_error")
REWARD_BY_LABEL = {
    "complete": 1.0,
    "incomplete": 0.1,
    "agent_error": 0.0,
}
DEFAULT_MODELS = ("openai/gpt-5.4-nano", "deepseek/deepseek-v4-flash")


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                yield line_no, json.loads(line)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def parse_json_maybe(value: Any, default: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return value if value is not None else default


def iter_tool_calls(completion: list[dict[str, Any]]):
    for message in completion:
        if message.get("role") != "assistant":
            continue
        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function") or {}
            yield {
                "name": function.get("name"),
                "arguments": parse_json_maybe(function.get("arguments"), {}),
            }


def final_answer(completion: list[dict[str, Any]]) -> str | None:
    for message in reversed(completion):
        if message.get("role") == "assistant" and not message.get("tool_calls"):
            content = message.get("content")
            return content if isinstance(content, str) else None
    return None


def task_from_rollout(rollout: dict[str, Any]) -> str | None:
    prompt = rollout.get("prompt") or []
    if prompt and isinstance(prompt[-1], dict):
        return prompt[-1].get("content")
    return None


def strip_judge_from_verify_result(verify_result: Any) -> tuple[Any, Any]:
    clean = copy.deepcopy(verify_result)
    original_judge = None
    if isinstance(clean, dict):
        original_judge = clean.pop("llm_judge", None)
        clean.pop("llm_judge_error", None)
    return clean, original_judge


def stable_payload_id(record: dict[str, Any]) -> str:
    key = json.dumps(
        {
            "scenario": record.get("scenario"),
            "task_idx": record.get("task_idx"),
            "trajectory": record.get("trajectory"),
            "sql_verifier": record.get("sql_verifier", {}).get("verify_result"),
        },
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def verifier_metadata(loader: AWMDataLoader, scenario: str, task_idx: int) -> dict[str, str]:
    entry = loader.get_verifier(scenario, task_idx, "sql") or {}
    raw_response_str = entry.get("verification", {}).get("raw_response", "{}")
    raw = parse_json_maybe(raw_response_str, {})
    if not isinstance(raw, dict):
        raw = {}
    return {
        "verifier_reasoning": raw.get("reasoning", ""),
        "success_criteria": raw.get("success_criteria", ""),
        "failure_criteria": raw.get("failure_criteria", ""),
    }


async def replay_rollout(
    rollout: dict[str, Any],
    verifier_mode: str,
    env_url: str,
    llm_base_url: str | None,
    llm_api_key: str | None,
    llm_model: str | None,
) -> dict[str, Any]:
    scenario = rollout["scenario"]
    task_idx = int(rollout["task_idx"])
    completion = rollout.get("completion") or []
    stats = Counter()
    replay_errors: list[str] = []

    async with AWMEnv(base_url=env_url) as env:
        reset = await env.reset(
            scenario=scenario,
            task_idx=task_idx,
            llm_base_url=llm_base_url,
            llm_api_key=llm_api_key,
            llm_model=llm_model,
        )
        task = reset.observation.task or task_from_rollout(rollout)

        for tool_call in iter_tool_calls(completion):
            name = tool_call["name"]
            arguments = parse_json_maybe(tool_call["arguments"], {})
            if not isinstance(arguments, dict):
                arguments = {}

            try:
                if name == "list_tools":
                    await env.step(ListToolsAction())
                    stats["list_tools"] += 1
                elif name == "call_tool":
                    tool_name = arguments.get("tool_name", "")
                    tool_args = parse_json_maybe(arguments.get("arguments"), {})
                    if not isinstance(tool_args, dict):
                        tool_args = {}
                    await env.step(CallToolAction(tool_name=tool_name, arguments=tool_args))
                    stats["call_tool"] += 1
                else:
                    stats[f"skipped:{name or 'missing'}"] += 1
            except Exception as exc:  # noqa: BLE001 - continue to verifier for audit.
                replay_errors.append(f"{name}: {type(exc).__name__}: {exc}")
                stats["replay_exception"] += 1

        verify_args = {"verifier_mode": verifier_mode}
        if verifier_mode == "code":
            answer = final_answer(completion)
            if answer:
                verify_args["final_answer"] = answer
        started = time.perf_counter()
        result = await env.step(CallToolAction(tool_name="verify", arguments=verify_args))
        elapsed = time.perf_counter() - started

        try:
            await env.step(CallToolAction(tool_name="done", arguments={"keep_session": False}))
        except Exception:
            pass

    verify_result = result.observation.verify_result
    if verifier_mode == "sql":
        verify_result, original_judge = strip_judge_from_verify_result(verify_result)
    else:
        original_judge = None

    return {
        "task": task,
        "reward_type": result.observation.reward_type,
        "reward": result.reward,
        "verify_result": verify_result,
        "original_judge": original_judge,
        "replay_stats": dict(stats),
        "replay_errors": replay_errors,
        "verify_seconds": elapsed,
    }


async def collect_payloads(args: argparse.Namespace) -> None:
    out = Path(args.output)
    if out.exists() and not args.append:
        out.unlink()

    llm_base_url = args.llm_base_url or os.environ.get("OPENENV_AWM_LLM_BASE_URL")
    llm_api_key = args.llm_api_key or os.environ.get("OPENENV_AWM_LLM_API_KEY")
    llm_model = args.llm_model or os.environ.get("OPENENV_AWM_LLM_MODEL")
    allowed = set(args.statuses.split(",")) if args.statuses else None
    loader = AWMDataLoader()

    written = 0
    seen = 0
    seen_tasks: set[tuple[str, int]] = set()
    for line_no, rollout in read_jsonl(Path(args.rollouts)):
        if allowed and rollout.get("status") not in allowed:
            continue

        scenario = rollout["scenario"]
        task_idx = int(rollout["task_idx"])
        task_key = (scenario, task_idx)
        if not args.all_rollouts:
            if task_key in seen_tasks:
                continue
            seen_tasks.add(task_key)

        seen += 1
        if args.limit and written >= args.limit:
            break
        if seen <= args.start:
            continue

        print(f"[collect] line={line_no} {scenario}#{task_idx}", flush=True)
        try:
            code = await replay_rollout(
                rollout, "code", args.env_url, llm_base_url, llm_api_key, llm_model
            )
            sql = await replay_rollout(
                rollout, "sql", args.env_url, llm_base_url, llm_api_key, llm_model
            )
            metadata = verifier_metadata(loader, scenario, task_idx)
            record = {
                "source_rollout_line": line_no,
                "scenario": scenario,
                "task_idx": task_idx,
                "task": sql.get("task") or code.get("task") or task_from_rollout(rollout),
                "source_status": rollout.get("status"),
                "source_reward": rollout.get("reward"),
                "prompt": rollout.get("prompt") or [],
                "trajectory": rollout.get("completion") or [],
                "code_verifier": code,
                "sql_verifier": sql,
                **metadata,
            }
            record["payload_id"] = stable_payload_id(record)
        except Exception as exc:  # noqa: BLE001 - preserve failed replay rows.
            record = {
                "source_rollout_line": line_no,
                "scenario": scenario,
                "task_idx": task_idx,
                "source_status": rollout.get("status"),
                "source_reward": rollout.get("reward"),
                "collection_error": f"{type(exc).__name__}: {exc}",
            }
        append_jsonl(out, record)
        written += 1

    print(f"[collect] wrote {written} records to {out}")


async def judge_payloads(args: argparse.Namespace) -> None:
    out = Path(args.output)
    if out.exists() and not args.append:
        out.unlink()

    models = args.models or list(DEFAULT_MODELS)
    llm_base_url = args.llm_base_url or os.environ.get("OPENENV_AWM_LLM_BASE_URL")
    llm_api_key = args.llm_api_key or os.environ.get("OPENENV_AWM_LLM_API_KEY")
    if not llm_base_url or not llm_api_key:
        raise SystemExit("Set --llm-base-url/--llm-api-key or OPENENV_AWM_LLM_* env vars")

    count = 0
    for _, payload in read_jsonl(Path(args.payloads)):
        if payload.get("collection_error"):
            continue
        verifier_result = payload.get("sql_verifier", {}).get("verify_result")
        if verifier_result is None:
            continue
        for model in models:
            for vote_idx in range(args.votes):
                started = time.perf_counter()
                classification, result = await run_llm_judge(
                    task=payload.get("task") or "",
                    verifier_result=verifier_result,
                    llm_base_url=llm_base_url,
                    llm_api_key=llm_api_key,
                    llm_model=model,
                    trajectory=payload.get("trajectory") or [],
                    verifier_reasoning=payload.get("verifier_reasoning", ""),
                    success_criteria=payload.get("success_criteria", ""),
                    failure_criteria=payload.get("failure_criteria", ""),
                )
                elapsed = time.perf_counter() - started
                append_jsonl(
                    out,
                    {
                        "payload_id": payload["payload_id"],
                        "scenario": payload.get("scenario"),
                        "task_idx": payload.get("task_idx"),
                        "model": model,
                        "vote_idx": vote_idx,
                        "classification": classification,
                        "reward": REWARD_BY_LABEL.get(classification),
                        "ok": classification in VALID_LABELS,
                        "judge_result": result,
                        "latency_s": elapsed,
                    },
                )
                count += 1
                print(
                    f"[judge] {count} payload={payload['payload_id'][:8]} "
                    f"model={model} vote={vote_idx} label={classification}",
                    flush=True,
                )

    print(f"[judge] wrote {count} votes to {out}")


def pairwise_agreement(groups: list[list[str]]) -> float | None:
    agree = 0
    total = 0
    for labels in groups:
        for i, left in enumerate(labels):
            for right in labels[i + 1 :]:
                total += 1
                agree += int(left == right)
    return agree / total if total else None


def fleiss_kappa(groups: list[list[str]], labels: tuple[str, ...]) -> float | None:
    usable = [g for g in groups if len(g) >= 2]
    if not usable:
        return None
    observed = []
    pooled = Counter()
    total_labels = 0
    for group in usable:
        counts = Counter(group)
        n = len(group)
        observed.append(sum(v * (v - 1) for v in counts.values()) / (n * (n - 1)))
        pooled.update(counts)
        total_labels += n
    p_e = sum((pooled[label] / total_labels) ** 2 for label in labels)
    p_bar = sum(observed) / len(observed)
    if abs(1 - p_e) < 1e-12:
        return None
    return (p_bar - p_e) / (1 - p_e)


def majority(values: list[str]) -> str | None:
    if not values:
        return None
    return Counter(values).most_common(1)[0][0]


def metrics_for_model(votes: list[dict[str, Any]]) -> dict[str, Any]:
    by_payload: dict[str, list[str]] = defaultdict(list)
    failures = Counter()
    for vote in votes:
        label = vote.get("classification")
        if label in VALID_LABELS:
            by_payload[vote["payload_id"]].append(label)
        else:
            failures[label or "missing"] += 1

    groups = list(by_payload.values())
    reward_groups = [
        [label for label in group if label in REWARD_BY_LABEL]
        for group in groups
    ]
    reward_groups = [group for group in reward_groups if len(group) >= 2]
    label_flip = sum(len(set(group)) > 1 for group in groups)
    reward_flip = sum(
        len({REWARD_BY_LABEL[label] for label in group}) > 1 for group in reward_groups
    )
    return {
        "payloads": len(groups),
        "votes": sum(len(group) for group in groups),
        "non_ok_votes": dict(failures),
        "pairwise_agreement": pairwise_agreement(groups),
        "fleiss_kappa": fleiss_kappa(groups, VALID_LABELS),
        "label_flip_rate": label_flip / len(groups) if groups else None,
        "reward_flip_rate": reward_flip / len(reward_groups) if reward_groups else None,
        "reward_flip_denominator": len(reward_groups),
    }


def write_metrics(args: argparse.Namespace) -> None:
    votes = [vote for _, vote in read_jsonl(Path(args.votes))]
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for vote in votes:
        by_model[vote["model"]].append(vote)

    metrics = {"models": {}, "cross_model": {}}
    for model, model_votes in sorted(by_model.items()):
        metrics["models"][model] = metrics_for_model(model_votes)

    if len(by_model) >= 2:
        majorities: dict[str, dict[str, str]] = defaultdict(dict)
        for model, model_votes in by_model.items():
            labels_by_payload: dict[str, list[str]] = defaultdict(list)
            for vote in model_votes:
                if vote.get("classification") in VALID_LABELS:
                    labels_by_payload[vote["payload_id"]].append(vote["classification"])
            for payload_id, labels in labels_by_payload.items():
                maj = majority(labels)
                if maj is not None:
                    majorities[payload_id][model] = maj
        model_names = sorted(by_model)
        pairs = []
        for left, right in combinations(model_names, 2):
            common = [
                payload_majorities
                for payload_majorities in majorities.values()
                if left in payload_majorities and right in payload_majorities
            ]
            same_label = sum(m[left] == m[right] for m in common)
            same_reward = sum(
                REWARD_BY_LABEL.get(m[left], "infra") == REWARD_BY_LABEL.get(m[right], "infra")
                for m in common
            )
            pairs.append(
                {
                    "models": [left, right],
                    "common_payloads": len(common),
                    "majority_label_agreement": same_label / len(common) if common else None,
                    "majority_reward_agreement": same_reward / len(common) if common else None,
                }
            )
        metrics["cross_model"] = {
            "models": model_names,
            "pairs": pairs,
        }

    metrics_path = Path(args.metrics_json)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    lines = ["# AWM Judge Reliability", ""]
    for model, data in metrics["models"].items():
        lines.extend(
            [
                f"## {model}",
                "",
                f"- Payloads: {data['payloads']}",
                f"- Votes: {data['votes']}",
                f"- Pairwise agreement: {fmt(data['pairwise_agreement'])}",
                f"- Fleiss kappa: {fmt(data['fleiss_kappa'])}",
                f"- Label-flip rate: {fmt(data['label_flip_rate'])}",
                f"- Reward-flip rate: {fmt(data['reward_flip_rate'])}",
                f"- Non-ok votes: {data['non_ok_votes']}",
                "",
            ]
        )
    if metrics["cross_model"]:
        cross = metrics["cross_model"]
        lines.extend(
            [
                "## Cross Model",
                "",
                f"- Models: {', '.join(cross['models'])}",
                "",
            ]
        )
        for pair in cross["pairs"]:
            lines.extend(
                [
                    f"### {' vs '.join(pair['models'])}",
                    "",
                    f"- Common payloads: {pair['common_payloads']}",
                    f"- Majority-label agreement: {fmt(pair['majority_label_agreement'])}",
                    f"- Majority-reward agreement: {fmt(pair['majority_reward_agreement'])}",
                    "",
                ]
            )
    summary_path = Path(args.summary_md)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[metrics] wrote {metrics_path} and {args.summary_md}")


def fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    collect = sub.add_parser("collect", help="Replay rollouts and save verifier payloads.")
    collect.add_argument("--rollouts", required=True)
    collect.add_argument("--output", default="judge_payloads.jsonl")
    collect.add_argument("--env-url", default=os.environ.get("AWM_BASE_URL", "http://localhost:8899"))
    collect.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Maximum unique tasks to collect; 0 means all matching tasks.",
    )
    collect.add_argument(
        "--start", type=int, default=0, help="Skip this many matching unique tasks."
    )
    collect.add_argument(
        "--all-rollouts",
        action="store_true",
        help=(
            "Collect every matching rollout instead of only the first rollout for "
            "each (scenario, task_idx); --limit and --start then count rollouts."
        ),
    )
    collect.add_argument(
        "--statuses",
        default="complete,incomplete,agent_error",
        help="Comma-separated rollout statuses to replay. Empty means all.",
    )
    collect.add_argument("--llm-base-url", default=None)
    collect.add_argument("--llm-api-key", default=None)
    collect.add_argument("--llm-model", default=None)
    collect.add_argument("--append", action="store_true")

    judge = sub.add_parser("judge", help="Run repeated judge votes from frozen payloads.")
    judge.add_argument("--payloads", required=True)
    judge.add_argument("--output", default="judge_votes.jsonl")
    judge.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    judge.add_argument("--votes", type=int, default=5)
    judge.add_argument("--llm-base-url", default=None)
    judge.add_argument("--llm-api-key", default=None)
    judge.add_argument("--append", action="store_true")

    metrics = sub.add_parser("metrics", help="Compute reliability metrics from judge votes.")
    metrics.add_argument("--votes", required=True)
    metrics.add_argument("--metrics-json", default="judge_reliability_metrics.json")
    metrics.add_argument("--summary-md", default="judge_reliability_summary.md")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "collect":
        asyncio.run(collect_payloads(args))
    elif args.command == "judge":
        asyncio.run(judge_payloads(args))
    elif args.command == "metrics":
        write_metrics(args)
    else:
        raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
