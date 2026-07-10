"""Collect verified Agent World Model demonstrations for supervised fine-tuning.

The collector uses the same two wrapper tools and sliding conversation window as
``async_grpo_awm.py``. Every attempt is appended to ``episodes.jsonl`` for
auditability and resume. Verified successes are deterministically expanded into
per-assistant-turn ``prompt``/``completion`` rows in ``sft.jsonl``.

Examples:
    uv run python -m awm.sft_dataset_awm collect --output-dir awm-sft-data
    uv run python -m awm.sft_dataset_awm validate --output-dir awm-sft-data
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable, Iterator
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
from typing import Any

try:
    from .trajectory_utils import windowed_messages
except ImportError:  # Direct execution: ``python awm/sft_dataset_awm.py``
    from trajectory_utils import windowed_messages


SCHEMA_VERSION = 1
EPISODES_FILENAME = "episodes.jsonl"
SFT_FILENAME = "sft.jsonl"
METADATA_FILENAME = "metadata.json"

WRAPPER_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_tools",
            "description": "Discover all MCP tools available for the current task. Call this first.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "call_tool",
            "description": "Invoke one MCP tool returned by list_tools.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tool_name": {
                        "type": "string",
                        "description": "Exact MCP tool name returned by list_tools.",
                    },
                    "arguments": {
                        "type": "object",
                        "description": "Arguments for the MCP tool.",
                        "additionalProperties": True,
                    },
                },
                "required": ["tool_name", "arguments"],
                "additionalProperties": False,
            },
        },
    },
]

_WRAPPER_TOOL_NAMES = {tool["function"]["name"] for tool in WRAPPER_TOOLS}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _record_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _task_key(value: dict[str, Any]) -> tuple[str, int]:
    return str(value["scenario"]), int(value["task_idx"])


def _episode_id(scenario: str, task_idx: int, attempt: int) -> str:
    return f"{scenario}#{task_idx}@{attempt}"


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield line_number, value


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _dump_model(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    if isinstance(value, dict):
        return deepcopy(value)
    raise TypeError(f"Unsupported assistant message type: {type(value).__name__}")


def normalize_assistant_message(
    value: Any,
    *,
    fallback_call_id_prefix: str,
) -> tuple[dict[str, Any], list[str]]:
    """Normalize an OpenAI assistant message to TRL's tool-call schema."""
    raw = _dump_model(value)
    output: dict[str, Any] = {"role": "assistant"}
    errors: list[str] = []

    if raw.get("content") is not None:
        output["content"] = raw["content"]

    calls: list[dict[str, Any]] = []
    for index, call in enumerate(raw.get("tool_calls") or []):
        if not isinstance(call, dict):
            errors.append(f"tool_call[{index}] is not an object")
            continue
        function = call.get("function") or {}
        name = function.get("name")
        if not isinstance(name, str) or not name:
            errors.append(f"tool_call[{index}] has no function name")
            name = ""

        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                errors.append(f"tool_call[{index}] arguments are not valid JSON")
                arguments = {}
        if not isinstance(arguments, dict):
            errors.append(f"tool_call[{index}] arguments are not an object")
            arguments = {}

        call_id = call.get("id")
        if not isinstance(call_id, str) or not call_id:
            call_id = f"{fallback_call_id_prefix}-{index}"
            errors.append(f"tool_call[{index}] has no id")
        calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        )

    if calls:
        output["tool_calls"] = calls
    return output, errors


def to_openai_messages(messages: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert normalized TRL messages back into Chat Completions parameters."""
    converted: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role == "assistant" and message.get("tool_calls"):
            assistant: dict[str, Any] = {"role": "assistant"}
            if "content" in message:
                assistant["content"] = message["content"]
            assistant["tool_calls"] = []
            for call in message["tool_calls"]:
                function = call["function"]
                assistant["tool_calls"].append(
                    {
                        "id": call["id"],
                        "type": "function",
                        "function": {
                            "name": function["name"],
                            "arguments": json.dumps(
                                function.get("arguments", {}), ensure_ascii=False
                            ),
                        },
                    }
                )
            converted.append(assistant)
        elif role == "tool":
            converted.append(
                {
                    "role": "tool",
                    "tool_call_id": message["tool_call_id"],
                    "content": str(message.get("content", "")),
                }
            )
        else:
            converted.append(deepcopy(message))
    return converted


def _usage_dict(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump(exclude_none=True)
    if isinstance(usage, dict):
        return deepcopy(usage)
    return {}


def _validate_outer_call(call: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    function = call.get("function") or {}
    name = function.get("name")
    arguments = function.get("arguments")
    if name not in _WRAPPER_TOOL_NAMES:
        errors.append(f"unknown outer tool {name!r}")
        return errors
    if not isinstance(arguments, dict):
        errors.append(f"{name} arguments are not an object")
        return errors
    if name == "list_tools" and arguments:
        errors.append("list_tools received unexpected arguments")
    if name == "call_tool":
        if (
            not isinstance(arguments.get("tool_name"), str)
            or not arguments["tool_name"]
        ):
            errors.append("call_tool.tool_name must be a non-empty string")
        if not isinstance(arguments.get("arguments"), dict):
            errors.append("call_tool.arguments must be an object")
    return errors


async def run_teacher_episode(
    task: dict[str, Any],
    attempt: int,
    *,
    client: Any,
    model: str,
    env_url: str,
    verifier_mode: str,
    max_turns: int,
    max_completion_tokens: int,
    temperature: float,
    context_window_turns: int,
) -> dict[str, Any]:
    """Run and score one teacher attempt against an isolated AWM session."""
    try:
        from . import async_grpo_awm as awm_training
    except ImportError:  # Direct execution
        import async_grpo_awm as awm_training

    scenario, task_idx = _task_key(task)
    episode_id = _episode_id(scenario, task_idx, attempt)
    prompt = deepcopy(task["prompt"])
    completion: list[dict[str, Any]] = []
    turns: list[dict[str, Any]] = []
    structural_errors: list[str] = []
    usage = Counter()
    first_call_is_list_tools = False
    truncated = False
    reached_turn_limit = False
    reward = 0.0
    status = "collector_error"
    env = awm_training.AWMEnvironment(env_url)
    scored = False

    try:
        await env.reset(
            scenario=scenario, task_idx=task_idx, verifier_mode=verifier_mode
        )
        for turn_index in range(max_turns):
            context = windowed_messages(prompt, completion, context_window_turns)
            response = await client.chat.completions.create(
                model=model,
                messages=to_openai_messages(context),
                tools=WRAPPER_TOOLS,
                parallel_tool_calls=False,
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
            )
            choice = response.choices[0]
            assistant, normalization_errors = normalize_assistant_message(
                choice.message,
                fallback_call_id_prefix=f"{episode_id}-turn-{turn_index}",
            )
            structural_errors.extend(normalization_errors)
            completion_index = len(completion)
            completion.append(assistant)

            for key, value in _usage_dict(getattr(response, "usage", None)).items():
                if isinstance(value, int):
                    usage[key] += value

            finish_reason = getattr(choice, "finish_reason", None)
            calls = assistant.get("tool_calls") or []
            turn_record = {
                "turn_index": turn_index,
                "completion_index": completion_index,
                "finish_reason": finish_reason,
                "trainable": False,
                "target_reason": None,
            }
            turns.append(turn_record)

            if finish_reason == "length":
                truncated = True
                structural_errors.append(
                    f"turn {turn_index} hit the completion token limit"
                )
                turn_record["target_reason"] = "truncated"
                break

            if not calls:
                content = assistant.get("content")
                if isinstance(content, str) and content.strip():
                    turn_record["trainable"] = True
                    turn_record["target_reason"] = "final_answer"
                else:
                    structural_errors.append(
                        f"turn {turn_index} ended without content or a tool call"
                    )
                    turn_record["target_reason"] = "empty_assistant"
                break

            if turn_index == 0:
                first_call_is_list_tools = (
                    len(calls) == 1
                    and calls[0].get("function", {}).get("name") == "list_tools"
                )
                if not first_call_is_list_tools:
                    structural_errors.append(
                        "first assistant turn did not call list_tools exactly once"
                    )

            if len(calls) != 1:
                structural_errors.append(
                    f"turn {turn_index} emitted {len(calls)} tool calls"
                )
                turn_record["target_reason"] = "parallel_tool_calls"
                for call in calls:
                    completion.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "name": call.get("function", {}).get("name", ""),
                            "content": "Error: exactly one wrapper tool call is allowed per turn.",
                        }
                    )
                break

            call = calls[0]
            call_errors = _validate_outer_call(call)
            structural_errors.extend(
                f"turn {turn_index}: {error}" for error in call_errors
            )
            function = call["function"]
            name = function["name"]
            arguments = function.get("arguments") or {}
            reward_type = "outer_tool_error"

            if call_errors:
                tool_response = "Error: " + "; ".join(call_errors)
            elif name == "list_tools":
                tool_response = await env.list_tools()
                reward_type = env.last_tool_reward_type or "tool_call_ok"
            else:
                tool_response = await env.call_tool(
                    tool_name=arguments["tool_name"],
                    arguments=arguments["arguments"],
                )
                reward_type = env.last_tool_reward_type or "unknown"

            completion.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": name,
                    "content": str(tool_response),
                }
            )
            turn_record["tool_reward_type"] = reward_type
            if not call_errors and reward_type == "tool_call_ok":
                turn_record["trainable"] = True
                turn_record["target_reason"] = "successful_tool_call"
            else:
                turn_record["target_reason"] = f"tool_error:{reward_type}"
        else:
            reached_turn_limit = True
            structural_errors.append(f"episode reached max_turns={max_turns}")

        reward, status = await env._score_rollout()
        scored = True
    except Exception as exc:
        status = f"collector_error:{type(exc).__name__}"
        structural_errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        if not scored:
            try:
                await env._close_session()
            except Exception:
                pass
        try:
            await env.close()
        except Exception:
            pass

    eligible = (
        status == "complete"
        and abs(float(reward) - 1.0) < 1e-9
        and first_call_is_list_tools
        and not structural_errors
        and not truncated
        and not reached_turn_limit
        and any(turn["trainable"] for turn in turns)
    )
    if eligible:
        eligibility_reason = "verified_complete"
    elif status != "complete" or abs(float(reward) - 1.0) >= 1e-9:
        eligibility_reason = f"verifier:{status}:reward={reward}"
    elif structural_errors:
        eligibility_reason = "structural_error"
    else:
        eligibility_reason = "not_trainable"

    return {
        "schema_version": SCHEMA_VERSION,
        "episode_id": episode_id,
        "scenario": scenario,
        "task_idx": task_idx,
        "task": task.get("task") or prompt[-1]["content"],
        "attempt": attempt,
        "reward": float(reward),
        "status": status,
        "eligible": eligible,
        "eligibility_reason": eligibility_reason,
        "structural_errors": structural_errors,
        "messages": [*prompt, *completion],
        "tools": deepcopy(WRAPPER_TOOLS),
        "turns": turns,
        "metrics": {
            "assistant_turns": len(turns),
            "tool_calls": sum(
                bool(message.get("tool_calls")) for message in completion
            ),
            "truncated": truncated,
            "reached_turn_limit": reached_turn_limit,
            "usage": dict(usage),
        },
        "collected_at": _utc_now(),
    }


def derive_sft_rows(
    record: dict[str, Any], context_window_turns: int
) -> list[dict[str, Any]]:
    """Expand one eligible episode into target-only conversational SFT rows."""
    if not record.get("eligible"):
        return []
    messages = record["messages"]
    prompt = deepcopy(messages[:2])
    completion = messages[2:]
    rows: list[dict[str, Any]] = []
    for turn in record.get("turns", []):
        if not turn.get("trainable"):
            continue
        completion_index = int(turn["completion_index"])
        if completion_index >= len(completion):
            raise ValueError(
                f"{record['episode_id']}: completion_index {completion_index} is out of range"
            )
        target = completion[completion_index]
        if target.get("role") != "assistant":
            raise ValueError(
                f"{record['episode_id']}: completion_index {completion_index} is not assistant"
            )
        context = windowed_messages(
            prompt, completion[:completion_index], context_window_turns
        )
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "id": f"{record['episode_id']}/turn-{turn['turn_index']}",
                "episode_id": record["episode_id"],
                "scenario": record["scenario"],
                "task_idx": record["task_idx"],
                "turn_index": turn["turn_index"],
                "prompt": deepcopy(context),
                "completion": [deepcopy(target)],
                "tools": deepcopy(record.get("tools") or WRAPPER_TOOLS),
            }
        )
    return rows


def rebuild_sft_dataset(
    episodes_path: Path,
    sft_path: Path,
    context_window_turns: int,
) -> tuple[int, int]:
    """Rewrite ``sft.jsonl`` from the first eligible success for each task."""
    tmp = sft_path.with_suffix(sft_path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    successful_tasks: set[tuple[str, int]] = set()
    row_count = 0
    with tmp.open("w", encoding="utf-8") as output:
        for _, record in iter_jsonl(episodes_path):
            key = _task_key(record)
            if key in successful_tasks or not record.get("eligible"):
                continue
            successful_tasks.add(key)
            for row in derive_sft_rows(record, context_window_turns):
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                row_count += 1
    tmp.replace(sft_path)
    return len(successful_tasks), row_count


def load_resume_state(
    episodes_path: Path,
) -> tuple[dict[tuple[str, int], int], set[tuple[str, int]]]:
    attempts: dict[tuple[str, int], int] = {}
    successes: set[tuple[str, int]] = set()
    for _, record in iter_jsonl(episodes_path):
        key = _task_key(record)
        attempts[key] = max(attempts.get(key, 0), int(record.get("attempt", 0)))
        if record.get("eligible"):
            successes.add(key)
    return attempts, successes


@dataclass
class CollectionProgress:
    attempted: int = 0
    succeeded: int = 0
    exhausted: int = 0


async def collect_task_records(
    tasks: list[dict[str, Any]],
    *,
    previous_attempts: dict[tuple[str, int], int],
    successful_tasks: set[tuple[str, int]],
    max_attempts: int,
    concurrency: int,
    run_attempt: Callable[[dict[str, Any], int], Awaitable[dict[str, Any]]],
    on_record: Callable[[dict[str, Any]], Awaitable[None]],
) -> CollectionProgress:
    """Collect tasks concurrently while keeping each task's retries sequential."""
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    progress = CollectionProgress()
    progress_lock = asyncio.Lock()

    pending = [task for task in tasks if _task_key(task) not in successful_tasks]
    for task in pending:
        queue.put_nowait(task)
    worker_count = min(concurrency, len(pending))
    for _ in range(worker_count):
        queue.put_nowait(None)

    async def worker() -> None:
        while True:
            task = await queue.get()
            try:
                if task is None:
                    return
                key = _task_key(task)
                start = previous_attempts.get(key, 0) + 1
                task_succeeded = False
                for attempt in range(start, max_attempts + 1):
                    record = await run_attempt(task, attempt)
                    await on_record(record)
                    async with progress_lock:
                        progress.attempted += 1
                    if record.get("eligible"):
                        task_succeeded = True
                        successful_tasks.add(key)
                        async with progress_lock:
                            progress.succeeded += 1
                        break
                if not task_succeeded:
                    async with progress_lock:
                        progress.exhausted += 1
            finally:
                queue.task_done()

    if worker_count:
        await asyncio.gather(*(worker() for _ in range(worker_count)))
    return progress


def _tool_names(tools: Any) -> list[str]:
    if not isinstance(tools, list):
        return []
    names = []
    for tool in tools:
        try:
            names.append(tool["function"]["name"])
        except (KeyError, TypeError):
            return []
    return names


def validate_sft_row(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(row.get("id"), str) or not row["id"]:
        errors.append("missing id")
    prompt = row.get("prompt")
    completion = row.get("completion")
    if not isinstance(prompt, list) or len(prompt) < 2:
        errors.append("prompt must contain at least system and user messages")
    if not isinstance(completion, list) or len(completion) != 1:
        errors.append("completion must contain exactly one message")
    elif completion[0].get("role") != "assistant":
        errors.append("completion target must have role=assistant")
    else:
        calls = completion[0].get("tool_calls") or []
        if len(calls) > 1:
            errors.append("assistant target contains parallel tool calls")
        for call in calls:
            errors.extend(_validate_outer_call(call))
            if not isinstance(call.get("id"), str) or not call["id"]:
                errors.append("assistant target tool call has no id")
    if _tool_names(row.get("tools")) != ["list_tools", "call_tool"]:
        errors.append("tools must contain list_tools and call_tool in canonical order")
    return errors


def validate_episode_structure(record: dict[str, Any]) -> list[str]:
    """Validate serialization invariants; failed episodes may record policy errors."""
    errors: list[str] = []
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return ["messages must contain at least system and user"]
    if [messages[0].get("role"), messages[1].get("role")] != ["system", "user"]:
        errors.append("messages must start with system then user")
    if _tool_names(record.get("tools")) != ["list_tools", "call_tool"]:
        errors.append("episode tools are not canonical wrapper tools")
    if not isinstance(record.get("turns"), list):
        errors.append("turns must be a list")
    if record.get("eligible"):
        if (
            record.get("status") != "complete"
            or abs(float(record.get("reward", 0)) - 1.0) >= 1e-9
        ):
            errors.append("eligible episode is not a reward-1 complete")
        if record.get("structural_errors"):
            errors.append("eligible episode contains structural_errors")
        cursor = 2
        assistant_number = 0
        saw_final_answer = False
        while cursor < len(messages):
            assistant = messages[cursor]
            if assistant.get("role") != "assistant":
                errors.append(f"messages[{cursor}] must be assistant")
                break
            calls = assistant.get("tool_calls") or []
            if not calls:
                saw_final_answer = True
                if cursor != len(messages) - 1:
                    errors.append("messages appear after the final assistant answer")
                break
            if len(calls) != 1:
                errors.append(f"messages[{cursor}] must contain exactly one tool call")
                break
            call = calls[0]
            errors.extend(
                f"messages[{cursor}]: {error}" for error in _validate_outer_call(call)
            )
            if (
                assistant_number == 0
                and call.get("function", {}).get("name") != "list_tools"
            ):
                errors.append("first assistant turn must call list_tools")
            if cursor + 1 >= len(messages):
                errors.append(f"messages[{cursor}] has no matching tool response")
                break
            tool_message = messages[cursor + 1]
            if tool_message.get("role") != "tool":
                errors.append(f"messages[{cursor + 1}] must be a tool response")
                break
            if tool_message.get("tool_call_id") != call.get("id"):
                errors.append(f"messages[{cursor + 1}] has the wrong tool_call_id")
            if tool_message.get("name") != call.get("function", {}).get("name"):
                errors.append(f"messages[{cursor + 1}] has the wrong tool name")
            cursor += 2
            assistant_number += 1
        if assistant_number == 0:
            errors.append("eligible episode contains no wrapper tool exchange")
        if saw_final_answer and not str(messages[-1].get("content", "")).strip():
            errors.append("final assistant answer is empty")
        completion_length = len(messages) - 2
        for turn in record.get("turns", []):
            index = turn.get("completion_index")
            if not isinstance(index, int) or not 0 <= index < completion_length:
                errors.append(f"invalid turn completion_index {index!r}")
            elif messages[index + 2].get("role") != "assistant":
                errors.append(
                    f"turn completion_index {index} does not reference assistant"
                )
        try:
            rows = derive_sft_rows(record, context_window_turns=3)
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(str(exc))
        else:
            if not rows:
                errors.append("eligible episode produces no SFT rows")
    return errors


def validate_dataset(
    output_dir: Path,
    *,
    tokenizer_id: str | None = None,
) -> dict[str, Any]:
    metadata_path = output_dir / METADATA_FILENAME
    episodes_path = output_dir / EPISODES_FILENAME
    sft_path = output_dir / SFT_FILENAME
    if not metadata_path.exists():
        raise ValueError(f"Missing {metadata_path}")
    if not episodes_path.exists():
        raise ValueError(f"Missing {episodes_path}")
    if not sft_path.exists():
        raise ValueError(f"Missing {sft_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    context_window_turns = int(metadata["config"]["context_window_turns"])

    expected_hashes: dict[str, str] = {}
    seen_tasks: set[tuple[str, int]] = set()
    episode_ids: set[str] = set()
    status_counts: Counter[str] = Counter()
    episode_count = 0
    errors: list[str] = []
    warnings: list[str] = []

    for line_number, record in iter_jsonl(episodes_path):
        episode_count += 1
        episode_id = record.get("episode_id")
        if episode_id in episode_ids:
            errors.append(
                f"{EPISODES_FILENAME}:{line_number}: duplicate episode_id {episode_id}"
            )
        elif isinstance(episode_id, str):
            episode_ids.add(episode_id)
        status_counts[str(record.get("status"))] += 1
        for error in validate_episode_structure(record):
            errors.append(f"{EPISODES_FILENAME}:{line_number}: {error}")
        key = _task_key(record)
        if record.get("eligible") and key not in seen_tasks:
            seen_tasks.add(key)
            try:
                rows = derive_sft_rows(record, context_window_turns)
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(f"{EPISODES_FILENAME}:{line_number}: {exc}")
            else:
                for row in rows:
                    expected_hashes[row["id"]] = _record_hash(row)

    sft_ids: set[str] = set()
    lengths: list[int] = []
    tokenizer = None
    if tokenizer_id:
        from transformers import AutoTokenizer
        from trl.chat_template_utils import qwen3_chat_template

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        tokenizer.chat_template = qwen3_chat_template

    for line_number, row in iter_jsonl(sft_path):
        row_id = row.get("id")
        if row_id in sft_ids:
            errors.append(f"{SFT_FILENAME}:{line_number}: duplicate id {row_id}")
        elif isinstance(row_id, str):
            sft_ids.add(row_id)
        for error in validate_sft_row(row):
            errors.append(f"{SFT_FILENAME}:{line_number}: {error}")
        expected = expected_hashes.get(row_id)
        if expected is None:
            errors.append(
                f"{SFT_FILENAME}:{line_number}: no eligible episode provenance for {row_id}"
            )
        elif expected != _record_hash(row):
            errors.append(
                f"{SFT_FILENAME}:{line_number}: row differs from deterministic derivation"
            )
        if tokenizer is not None:
            try:
                ids = tokenizer.apply_chat_template(
                    row["prompt"] + row["completion"],
                    tools=row["tools"],
                    tokenize=True,
                    add_generation_prompt=False,
                )
                lengths.append(len(ids))
            except Exception as exc:
                errors.append(
                    f"{SFT_FILENAME}:{line_number}: tokenizer render failed: {exc}"
                )

    missing = set(expected_hashes) - sft_ids
    if missing:
        errors.append(f"{len(missing)} deterministically derived SFT rows are missing")
    extra = sft_ids - set(expected_hashes)
    if extra:
        errors.append(f"{len(extra)} SFT rows have no eligible source episode")
    if not seen_tasks:
        warnings.append("dataset contains no eligible successful tasks")

    report: dict[str, Any] = {
        "valid": not errors,
        "episodes": episode_count,
        "successful_tasks": len(seen_tasks),
        "sft_rows": len(sft_ids),
        "status_counts": dict(status_counts),
        "errors": errors,
        "warnings": warnings,
    }
    if lengths:
        ordered = sorted(lengths)

        def percentile(p: float) -> int:
            return ordered[round((len(ordered) - 1) * p)]

        report["token_lengths"] = {
            "p50": percentile(0.50),
            "p95": percentile(0.95),
            "p99": percentile(0.99),
            "max": ordered[-1],
        }
    return report


def summarize_episodes(episodes_path: Path, selected_tasks: int) -> dict[str, Any]:
    statuses: Counter[str] = Counter()
    attempts = 0
    successes: set[tuple[str, int]] = set()
    attempted_tasks: set[tuple[str, int]] = set()
    turns: list[int] = []
    for _, record in iter_jsonl(episodes_path):
        attempts += 1
        key = _task_key(record)
        attempted_tasks.add(key)
        statuses[str(record.get("status"))] += 1
        turns.append(int(record.get("metrics", {}).get("assistant_turns", 0)))
        if record.get("eligible"):
            successes.add(key)
    return {
        "selected_tasks": selected_tasks,
        "attempted_tasks": len(attempted_tasks),
        "successful_tasks": len(successes),
        "unsuccessful_tasks": selected_tasks - len(successes),
        "attempts": attempts,
        "status_counts": dict(statuses),
        "mean_assistant_turns": statistics.fmean(turns) if turns else 0.0,
    }


def _load_tasks(
    args: argparse.Namespace, prompt_date: str
) -> tuple[list[dict[str, Any]], str]:
    try:
        from . import async_grpo_awm as awm_training
    except ImportError:
        import async_grpo_awm as awm_training

    dataset = awm_training.build_dataset(
        args.env_url,
        args.dataset_size,
        args.dataset_start,
        args.num_scenarios,
    )
    tasks: list[dict[str, Any]] = []
    system_prompt = awm_training.SYSTEM_PROMPT.format(today=prompt_date)
    for row in dataset:
        prompt = deepcopy(row["prompt"])
        prompt[0]["content"] = system_prompt
        tasks.append(
            {
                "scenario": row["scenario"],
                "task_idx": int(row["task_idx"]),
                "task": prompt[-1]["content"],
                "prompt": prompt,
            }
        )
    return tasks, system_prompt


def _collection_config(
    args: argparse.Namespace,
    *,
    prompt_date: str,
    system_prompt: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "env_url": args.env_url,
        "dataset_size": args.dataset_size,
        "dataset_start": args.dataset_start,
        "num_scenarios": args.num_scenarios,
        "shuffle_seed": 42,
        "prompt_date": prompt_date,
        "system_prompt_sha256": hashlib.sha256(system_prompt.encode()).hexdigest(),
        "teacher_base_url": args.teacher_base_url,
        "teacher_model": args.teacher_model,
        "verifier_mode": args.verifier_mode,
        "max_attempts": args.max_attempts,
        "concurrency": args.concurrency,
        "max_turns": args.max_turns,
        "max_completion_tokens": args.max_completion_tokens,
        "temperature": args.temperature,
        "context_window_turns": args.context_window_turns,
        "wrapper_tools_sha256": _record_hash(WRAPPER_TOOLS),
    }


async def collect_command(args: argparse.Namespace) -> None:
    if not args.teacher_base_url:
        raise ValueError(
            "Teacher base URL is required (--teacher-base-url or ENDPOINT_URL)"
        )
    if not args.teacher_model:
        raise ValueError(
            "Teacher model is required (--teacher-model or AWM_EXAMPLE_AGENT_MODEL)"
        )
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get(
        "OPENENV_AWM_LLM_API_KEY"
    )
    if not api_key:
        raise ValueError(
            "Set OPENAI_API_KEY or OPENENV_AWM_LLM_API_KEY for the teacher"
        )
    if args.dataset_size < 1:
        raise ValueError("dataset-size must be at least 1")
    if args.dataset_start < 0:
        raise ValueError("dataset-start must be non-negative")
    if args.num_scenarios is not None and args.num_scenarios < 1:
        raise ValueError("num-scenarios must be at least 1")
    if args.max_attempts < 1:
        raise ValueError("max-attempts must be at least 1")
    if args.concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    if args.max_turns < 1:
        raise ValueError("max-turns must be at least 1")
    if args.max_completion_tokens < 1:
        raise ValueError("max-completion-tokens must be at least 1")
    if args.context_window_turns < 0:
        raise ValueError("context-window-turns must be non-negative")

    output_dir = args.output_dir
    metadata_path = output_dir / METADATA_FILENAME
    episodes_path = output_dir / EPISODES_FILENAME
    sft_path = output_dir / SFT_FILENAME
    existing_metadata = None
    if metadata_path.exists():
        existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    if existing_metadata and args.resume:
        prompt_date = args.prompt_date or existing_metadata["config"]["prompt_date"]
    else:
        prompt_date = args.prompt_date or date.today().isoformat()
    try:
        date.fromisoformat(prompt_date)
    except ValueError as exc:
        raise ValueError("prompt-date must be an ISO date (YYYY-MM-DD)") from exc

    tasks, system_prompt = _load_tasks(args, prompt_date)
    config = _collection_config(
        args, prompt_date=prompt_date, system_prompt=system_prompt
    )

    if existing_metadata:
        if not args.resume:
            raise ValueError(
                f"{output_dir} already contains metadata; use --resume or a new directory"
            )
        if existing_metadata.get("config") != config:
            raise ValueError(
                "Resume configuration differs from metadata.json; use a new output directory"
            )
    elif episodes_path.exists() or sft_path.exists():
        raise ValueError(
            f"{output_dir} contains dataset files without compatible metadata.json"
        )
    else:
        _write_json(
            metadata_path,
            {"created_at": _utc_now(), "config": config, "summary": {}},
        )

    previous_attempts, successful_tasks = load_resume_state(episodes_path)

    try:
        from . import async_grpo_awm as awm_training
    except ImportError:
        import async_grpo_awm as awm_training
    awm_training._RESET_SEMAPHORE = asyncio.Semaphore(args.concurrency)

    from openai import AsyncOpenAI

    write_lock = asyncio.Lock()

    async def on_record(record: dict[str, Any]) -> None:
        async with write_lock:
            append_jsonl(episodes_path, record)
            print(
                f"[{record['episode_id']}] status={record['status']} "
                f"reward={record['reward']:.3f} eligible={record['eligible']}"
            )

    async with AsyncOpenAI(base_url=args.teacher_base_url, api_key=api_key) as client:

        async def run_attempt(task: dict[str, Any], attempt: int) -> dict[str, Any]:
            return await run_teacher_episode(
                task,
                attempt,
                client=client,
                model=args.teacher_model,
                env_url=args.env_url,
                verifier_mode=args.verifier_mode,
                max_turns=args.max_turns,
                max_completion_tokens=args.max_completion_tokens,
                temperature=args.temperature,
                context_window_turns=args.context_window_turns,
            )

        await collect_task_records(
            tasks,
            previous_attempts=previous_attempts,
            successful_tasks=successful_tasks,
            max_attempts=args.max_attempts,
            concurrency=args.concurrency,
            run_attempt=run_attempt,
            on_record=on_record,
        )

    successful_count, sft_rows = rebuild_sft_dataset(
        episodes_path,
        sft_path,
        args.context_window_turns,
    )
    summary = summarize_episodes(episodes_path, len(tasks))
    summary["sft_rows"] = sft_rows
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["updated_at"] = _utc_now()
    metadata["summary"] = summary
    _write_json(metadata_path, metadata)

    report = validate_dataset(output_dir)
    if not report["valid"]:
        raise RuntimeError(
            "Dataset validation failed:\n" + "\n".join(report["errors"][:20])
        )
    print(
        f"Collected {successful_count}/{len(tasks)} verified tasks -> "
        f"{sft_rows} SFT rows in {sft_path}"
    )


def validate_command(args: argparse.Namespace) -> None:
    metadata_path = args.output_dir / METADATA_FILENAME
    if args.rebuild:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        rebuild_sft_dataset(
            args.output_dir / EPISODES_FILENAME,
            args.output_dir / SFT_FILENAME,
            int(metadata["config"]["context_window_turns"]),
        )
    report = validate_dataset(args.output_dir, tokenizer_id=args.tokenizer)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["valid"]:
        raise SystemExit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser(
        "collect", help="Collect verified teacher demonstrations."
    )
    collect.add_argument("--env-url", default="http://localhost:8899")
    collect.add_argument("--teacher-base-url", default=os.environ.get("ENDPOINT_URL"))
    collect.add_argument(
        "--teacher-model", default=os.environ.get("AWM_EXAMPLE_AGENT_MODEL")
    )
    collect.add_argument("--output-dir", type=Path, default=Path("awm-sft-data"))
    collect.add_argument("--dataset-size", type=int, default=100)
    collect.add_argument("--dataset-start", type=int, default=0)
    collect.add_argument("--num-scenarios", type=int)
    collect.add_argument("--max-attempts", type=int, default=3)
    collect.add_argument("--concurrency", type=int, default=16)
    collect.add_argument("--max-turns", type=int, default=20)
    collect.add_argument("--max-completion-tokens", type=int, default=4096)
    collect.add_argument("--temperature", type=float, default=1.0)
    collect.add_argument("--context-window-turns", type=int, default=3)
    collect.add_argument("--verifier-mode", choices=["sql", "code"], default="sql")
    collect.add_argument(
        "--prompt-date", help="ISO date embedded in the system prompt."
    )
    collect.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume compatible append-only collection (default: true).",
    )
    collect.set_defaults(func=lambda args: asyncio.run(collect_command(args)))

    validate = subparsers.add_parser(
        "validate", help="Validate or rebuild a collected dataset."
    )
    validate.add_argument("--output-dir", type=Path, default=Path("awm-sft-data"))
    validate.add_argument(
        "--tokenizer", help="Optional tokenizer id for rendering and token stats."
    )
    validate.add_argument(
        "--rebuild", action="store_true", help="Rebuild sft.jsonl before validation."
    )
    validate.set_defaults(func=validate_command)
    return parser


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
