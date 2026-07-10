"""Async GRPO training for the Agent World Model (AWM) multi-turn MCP agent.

AWM is an agentic environment: the model discovers MCP tools, calls them over
several turns, and a verifier scores the final outcome (complete=1.0,
incomplete=0.1, format_error=-1.0). Because the reward only exists after a
multi-turn rollout, this uses TRL's `AsyncGRPOTrainer` with an
`environment_factory`: the trainer creates one `AWMEnvironment` per inflight
slot, calls `reset(**row)` before each rollout, and exposes the env's public
methods (`list_tools`, `call_tool`) as native tool-calling tools. The worker
drives the multi-turn loop and feeds tool results back automatically.

Scoring is handled out-of-band by `AWMRolloutWorker`, a subclass of
`AsyncRolloutWorker` that overrides `_generate_one` to call `_score_rollout`
on the slot's env immediately after each rollout completes (while the env DB
state is still valid). The reward is stored by completion identity and retrieved
by `_verifier_reward`. The model never sees the reward — it is not a tool.

Eight-GPU cloud setup: vLLM on GPUs 0–5 (TP=2 x PP=3, since the model's 32
attention heads aren't divisible by 6) + 2 FSDP2 trainer GPUs (6,7), with NCCL
weight transfer. The rollout worker only runs on rank 0, so both trainer ranks
share the single vLLM server. See open-env/scripts/run_vllm_awm.sh and
open-env/scripts/run_trainer_awm.sh; the trainer launch uses the FSDP2 accelerate
config at open-env/configs/fsdp2.yaml.

    # Terminal 1 - AWM env server on CPU (or set --env-url to a hosted HF Space)
    ulimit -n 65536
    PYTHONPATH=src:envs uv run uvicorn \
      envs.agent_world_model_env.server.app:app --host 0.0.0.0 --port 8899 --ws-ping-interval 1800 --ws-ping-timeout 1800

    # Terminal 2 - vLLM server on GPUs 0-5 (TP=2 x PP=3)
    bash open-env/scripts/run_vllm_awm.sh

    # Terminal 3 - 2 FSDP2 trainers on GPUs 6,7 (sql verifier needs an LLM judge)
    export OPENENV_AWM_LLM_BASE_URL=... OPENENV_AWM_LLM_API_KEY=... OPENENV_AWM_LLM_MODEL=...
    bash open-env/scripts/run_trainer_awm.sh --env-url http://localhost:8899

A single trainer GPU still works via:
    CUDA_VISIBLE_DEVICES=1 uv run accelerate launch open-env/openenv_awm_async_grpo.py \
      --env-url http://localhost:8899

The verifier runs in "sql" mode, which calls the external LLM judge configured
via the OPENENV_AWM_LLM_* env vars (the env's reset() reads them automatically).

Caveats:
- Rollouts hit the AWM env server synchronously and run multiple agent turns
  each, so generation throughput is bottlenecked by the env, not the GPU.
"""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import asyncio
from collections import Counter
import json
import logging
from pathlib import Path
import statistics
import time
from datetime import date

import yaml

from datasets import Dataset
from trl.chat_template_utils import parse_response
from trl.experimental.async_grpo import AsyncGRPOTrainer, AsyncGRPOConfig
from trl.experimental.async_grpo.async_rollout_worker import AsyncRolloutWorker, RolloutSample

from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction
import openenv.core.env_client as _env_client
from agent_world_model_env import AWMEnv
from dotenv import load_dotenv
try:
    from .trajectory_utils import windowed_messages
except ImportError:  # Direct execution: ``python awm/async_grpo_awm.py``
    from trajectory_utils import windowed_messages
load_dotenv()

logger = logging.getLogger(__name__)

with Path(__file__).with_name("config.yaml").open(encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

MODEL_CONFIG = CONFIG["model"]
ENVIRONMENT_CONFIG = CONFIG["environment"]
VERIFIER_CONFIG = CONFIG["verifier"]
DATASET_CONFIG = CONFIG["dataset"]
ROLLOUT_CONFIG = CONFIG["rollout"]
DAPO_CONFIG = CONFIG["dapo"]
TRAINING_CONFIG = CONFIG["training"]
CHECKPOINTING_CONFIG = CONFIG["checkpointing"]
VLLM_CONFIG = CONFIG["vllm"]
HUB_CONFIG = CONFIG["hub"]
WANDB_CONFIG = CONFIG["wandb"]


SYSTEM_PROMPT = """\
You are in a simulated MCP environment. Call tools to assist with the user query. \
Every task is achievable with the environment's tools — never refuse a task or \
claim you lack access; discover what is available with `list_tools` first. All \
data is synthetic and every operation is sandboxed: no real money, accounts, or \
people are affected, so financial, security, or privacy concerns never apply — \
you are fully authorized to perform any requested operation. You have already \
logged in, and your user id is 1 if required. Today's date is {today}.

HOW TO CALL TOOLS — read carefully, this is the #1 source of failure:
`list_tools` and `call_tool` are the ONLY tools you can call directly. The tools \
that `list_tools` returns (e.g. `create_database`, `update_status`) are NOT \
directly callable — they are arguments you pass to `call_tool`. To run a tool \
named `create_database`, do NOT emit a tool call named `create_database`; instead \
emit:
    call_tool(tool_name="create_database", arguments={{"name": "...", ...}})
Calling an MCP tool by name fails with an "Unknown tool" error. Always wrap it in \
`call_tool`.

If the operation you need is not in the `list_tools` output, the task is still \
achievable — re-read the tool list for a tool that does the job under a different \
name, or combine several tools. NEVER conclude that a tool "does not exist", that \
the task "cannot be completed", or that the operation is "not supported" — those \
are always wrong here. Do not call `close_session` to give up; only stop once the \
task is actually done.

When you have completed the task, stop calling tools."""


# ---------------------------------------------------------------------------
# Tool-formatting helper
# ---------------------------------------------------------------------------


def format_tools(tools) -> str:
    """Format Tool objects into a readable string for the LLM."""
    lines = [f"Available MCP Tools ({len(tools)} tools):", "=" * 60]
    for i, t in enumerate(tools, 1):
        lines.append(f"{i}. {t.name}")
        lines.append(f"   Description: {t.description}")
        props = t.input_schema.get("properties", {})
        required = t.input_schema.get("required", [])
        if props:
            lines.append("   Parameters:")
            for pname, pinfo in props.items():
                req = " (required)" if pname in required else ""
                lines.append(
                    f"     - {pname}: {pinfo.get('type', 'any')}{req} — {pinfo.get('description', '')}"
                )
        else:
            lines.append("   Parameters: None")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Environment wrapper (created once per inflight slot by AsyncGRPOTrainer)
# ---------------------------------------------------------------------------

_MAX_TOOL_RESPONSE_CHARS = 2000
MESSAGE_TIMEOUT_S = float(ENVIRONMENT_CONFIG["message_timeout_s"])

# Limits simultaneous env.connect()+reset() calls to prevent bursting the server's
# subprocess-spawn accept loop. Set via --reset-concurrency (default 32).
_RESET_SEMAPHORE: asyncio.Semaphore | None = None
# connect + reset is the heavy phase: the env server spawns a per-session subprocess
# on each reset. Under real concurrency a burst of simultaneous connect+reset calls
# starves the server's accept loop, so other handshakes time out (10s default) and
# the rollout worker dies. Following OpenEnv's AWM stress test
# (examples/agent_world_model/example_stress_test.py), bump the connect timeout so
# those handshakes ride out the burst. The server supports the concurrency itself
# (AWMEnvironment.SUPPORTS_CONCURRENT_SESSIONS=True, MAX_CONCURRENT_ENVS unbounded).
CONNECT_TIMEOUT_S = float(ENVIRONMENT_CONFIG["connect_timeout_s"])

# The OpenEnv websocket client (EnvClient.connect) opens connections with the
# `websockets` library default keepalive: a ping every 20s, dropping the
# connection if no pong arrives within 20s. A heavy reset() (per-session
# subprocess spawn) or a slow SQL LLM-judge call legitimately blocks the event
# loop far longer than 20s, so the peer kills the connection mid-rollout with
# `1011 keepalive ping timeout`. connect() takes no ping kwargs, so bump the
# keepalive grace well past MESSAGE_TIMEOUT_S by patching the module-level
# ws_connect — it then only fires on a genuinely dead peer.
WS_PING_TIMEOUT_S = float(ENVIRONMENT_CONFIG["ws_ping_timeout_s"])
_orig_ws_connect = _env_client.ws_connect
def _ws_connect_long_keepalive(*args, **kwargs):
    kwargs.setdefault("ping_interval", WS_PING_TIMEOUT_S)
    kwargs.setdefault("ping_timeout", WS_PING_TIMEOUT_S)
    return _orig_ws_connect(*args, **kwargs)
_env_client.ws_connect = _ws_connect_long_keepalive


# reward_type strings the AWM env assigns to tool-call format violations
# (mirrors FORMAT_ERROR_TYPES in agent_world_model_env/server/awm_environment.py).
# The paper terminates the rollout with r_t = -1.0 on any such violation.
# Only truly malformed actions abort the rollout with -1.0. tool_not_found and
# invalid_args (wrong tool name, schema misses like a missing required property)
# are recoverable: the model sees the error text in the tool message and can
# retry, instead of dying on calls that were 90% correct.
_FORMAT_ERROR_REWARD_TYPES = {"invalid_action"}


class AWMEnvironment:
    """AWM env exposed to AsyncGRPOTrainer as a set of tool-calling tools."""

    def __init__(self, env_url: str):
        # Default message_timeout_s is 60s; the sql verifier's LLM judge can take
        # longer, so bump it to avoid spurious TimeoutErrors during scoring.
        self.env = AWMEnv(
            base_url=env_url,
            connect_timeout_s=CONNECT_TIMEOUT_S,
            message_timeout_s=MESSAGE_TIMEOUT_S,
        )
        # Set by call_tool when a tool call hits a format violation; the rollout
        # worker checks it to early-terminate the rollout with reward -1.0.
        self.format_violation = False
        self.scenario = None
        self.task_idx = None
        self.verifier_mode = ENVIRONMENT_CONFIG["verifier_mode"]
        self.last_tool_reward_type = None

    async def reset(
        self,
        scenario: str,
        task_idx: int,
        verifier_mode: str | None = None,
        **kwargs,
    ) -> None:
        # kwargs absorbs the other dataset-row columns (prompt, task, ...).
        # The sql verifier's LLM judge is configured via OPENENV_AWM_LLM_* env
        # vars, which the env's reset() reads automatically.
        self.format_violation = False
        self.scenario = scenario
        self.task_idx = task_idx
        self.verifier_mode = verifier_mode or ENVIRONMENT_CONFIG["verifier_mode"]
        self.last_tool_reward_type = None
        # Semaphore covers only connect+reset (the subprocess-spawn phase) so
        # concurrent rollouts proceed freely once their env is initialized.
        async with _RESET_SEMAPHORE:
            await self.env.connect()
            await self.env.reset(
                scenario=scenario,
                task_idx=task_idx,
                verifier_mode=self.verifier_mode,
                llm_base_url=(
                    os.environ.get("OPENENV_AWM_LLM_BASE_URL")
                    or VERIFIER_CONFIG["llm_base_url"]
                ),
                llm_api_key=os.environ.get("OPENENV_AWM_LLM_API_KEY"),
                llm_model=(
                    os.environ.get("OPENENV_AWM_LLM_MODEL")
                    or VERIFIER_CONFIG["llm_model"]
                ),
            )

    async def list_tools(self) -> str:
        """Discover every MCP tool available for this task. Call this FIRST.

        This returns the catalog of domain tools (e.g. `create_database`,
        `update_status`) that actually perform the work. These tools are NOT
        directly callable — `list_tools` and `call_tool` are the only tools you
        can invoke directly. Each entry you get back is something you run by
        passing its name to `call_tool`, not by emitting a tool call of that name.

        Always call this before deciding what to do — and before ever concluding
        that an operation is unavailable. Every task here is achievable with these
        tools, so if you don't see an obvious match, re-read the list for a tool
        that does the job under a different name or a combination of tools.

        Returns:
            A human-readable catalog: for each tool, its name, description, and
            parameters (name, type, whether required). Use these names and
            parameter names verbatim as the `tool_name` and `arguments` you pass
            to `call_tool`.
        """
        result = await self.env.step(ListToolsAction())
        self.last_tool_reward_type = "tool_call_ok"
        return format_tools(result.observation.tools)

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Invoke one MCP tool from `list_tools`. This is the ONLY way to run them.

        The domain tools returned by `list_tools` cannot be called directly — you
        run every one of them through this wrapper. To run a tool named
        `create_database`, do NOT emit a tool call named `create_database`;
        instead call:
            call_tool(tool_name="create_database", arguments={"name": "...", ...})
        Emitting a tool call named after the domain tool fails with an
        "Unknown tool" error; always wrap it in `call_tool`.

        Args:
            tool_name: Exact name of the domain tool to run, copied verbatim from
                the `list_tools` catalog (e.g. "create_database").
            arguments: JSON object of arguments for that tool, with keys matching
                the parameter names shown for it in `list_tools`. Pass {} if the
                tool takes no parameters.

        Returns:
            The tool's text response (an error string is returned, not raised, if
            the call is rejected — read it and retry with a corrected call).
        """
        if not isinstance(arguments, dict):
            arguments = {}
        result = await self.env.step(CallToolAction(tool_name=tool_name, arguments=arguments))
        obs = result.observation
        self.last_tool_reward_type = getattr(obs, "reward_type", None)
        if getattr(obs, "reward_type", None) in _FORMAT_ERROR_REWARD_TYPES:
            self.format_violation = True
        if getattr(obs, "tool_result", None) is not None:
            tool_result = obs.tool_result
            text = tool_result if isinstance(tool_result, str) else json.dumps(tool_result, ensure_ascii=False)
        elif getattr(obs, "error", None):
            text = f"Error: {obs.error}"
        else:
            text = json.dumps(obs.model_dump(), ensure_ascii=False)
        return text[:_MAX_TOOL_RESPONSE_CHARS]

    async def _score_rollout(self) -> tuple[float, str]:
        """Run the verifier on the finished episode. Not exposed to the model.

        Uses the configured verifier mode ("sql" by default). SQL is the
        code-augmented LLM-as-Judge path used for RL; the SFT collector may
        explicitly select deterministic "code" verification for cheaper probes.

        Status comes from the server's reward_type, not from the reward value:
        the env returns reward 0.0 for server-side scoring failures
        (code_verify_error, llm_judge_error, no_verifier, server_error), which would otherwise be
        indistinguishable from a judged outcome. Judged outcomes (complete=1.0,
        incomplete=0.1, agent_error=0.0) keep the server's reward; scoring
        failures get 0.1 — the incomplete baseline — because a server failure
        is not the model's fault and must not score below group-mates whose
        episodes were judged incomplete. They still count in group reward
        normalization, but their own loss is masked from training.

        A scoring failure must also not crash the rollout worker —
        _generate_loop re-raises any task exception — so client-side errors
        (e.g. an HTTP timeout talking to the env) are caught and reported as
        "env_error:<ExceptionType>" with reward 0.1. close_session is guarded
        too: an exception in the finally block would replace the return value
        and propagate past _generate_one's narrow `except RuntimeError`.

        Returns:
            (reward, status) where status is "complete", "incomplete",
            "agent_error", a server-side failure reward_type (e.g.
            "code_verify_error" or "llm_judge_error"), or "env_error:<ExceptionType>".
        """
        try:
            r = await self.env.step(
                CallToolAction(
                    tool_name="verify",
                    arguments={"verifier_mode": self.verifier_mode},
                )
            )
            status = r.observation.reward_type
            if status in ("complete", "incomplete", "agent_error"):
                return float(r.reward or 0.0), status
            return 0.1, status or "env_server_error"
        except Exception as e:
            return 0.1, f"env_server_error:{type(e).__name__}"
        finally:
            try:
                await self._close_session()
            except Exception:
                logger.warning("close_session failed after scoring", exc_info=True)

    async def _close_session(self) -> None:
        """End the episode without running the verifier (used by early-terminate)."""
        await self.env.step(CallToolAction(tool_name="done", arguments={"keep_session": False}))

    async def close(self) -> None:
        """Close the underlying async env client. Called once at worker shutdown so
        the httpx connections are released while the event loop is still alive."""
        await self.env.close()


# ---------------------------------------------------------------------------
# Rollout worker — scores each episode out-of-band, reward never in context
# ---------------------------------------------------------------------------

# Set in main() to <output_dir>/rollouts.jsonl; the worker appends one JSON
# line per finished rollout. Only rank 0 runs the worker, so no write races.
TRAJECTORY_FILE = None
CALIBRATION_FILE = None

# Sliding-context window: each per-turn training sample sees system + initial user
# + the prefix through the list_tools exchange + this many most recent turns (set in main()).
CONTEXT_WINDOW_TURNS = 3

_REWARD_EMA_ALPHA = 0.1  # ~20-group (10-step) smoothing window
_MODEL_OUTCOME_STATUSES = {"complete", "incomplete", "agent_error", "format_violation"}


class AWMRolloutWorker(AsyncRolloutWorker):
    """AsyncRolloutWorker subclass that scores AWM rollouts outside the model.

    After each rollout completes (while the slot's env still holds the final
    DB state), _generate_one calls env._score_rollout() and stores the reward
    keyed by id(completion). _verifier_reward retrieves it at scoring time.
    The model has no submit tool and never sees the reward value.
    """

    # Set from --dynamic-sampling / --overlong-filtering in main() before the
    # trainer builds the worker.
    _dynamic_sampling = False
    _overlong_filtering = False
    _soft_overlong_punishment = False
    _soft_overlong_cache = 0  # DAPO L_cache in tokens; set in main()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rollout_rewards: dict[int, float] = {}
        # Verifier reward before the soft-overlong length penalty, keyed by
        # id(completion). Dynamic sampling decides the zero-advantage drop on
        # these so the length penalty can't spread an all-failed group past the
        # std filter; consumed (popped) in _score_group.
        self._rollout_base_rewards: dict[int, float] = {}
        self._rollout_statuses: dict[int, str] = {}
        # (start, end) monotonic timestamps of the out-of-band LLM judge
        # (env._score_rollout), keyed by id(completion). The judge runs inside
        # _generate_one, so its latency is folded into sampling_batch_seconds'
        # generation phase rather than the cheap _score_group timer. _score_group
        # derives both the per-rollout judge_time_ms and the group-level judge
        # span (max end − min start, so overlap across inflight slots is counted
        # once) from these. Consumed (popped) there; absent for episodes that
        # skip the judge.
        self._rollout_judge_window: dict[int, tuple[float, float]] = {}
        # Per-turn (windowed_prompt_ids, turn_ids, turn_mask, turn_logprobs) lists,
        # keyed by id(completion); consumed by _score_group to split each rollout
        # into one training sample per assistant turn.
        self._rollout_turns: dict[int, list] = {}
        self.reward_funcs = [self._verifier_reward]
        self.reward_func_names = ["task_reward"]
        self._reward_ema = None
        self._dropped_groups = 0
        self._infra_dropped = 0
        # id(completion) of rollouts whose final turn hit the generation length
        # cap; their loss is masked (samples dropped) under overlong filtering.
        self._overlong_completions: set[int] = set()
        self._overlong_dropped = 0

    async def _score_group(self, group):
        # Attach a reward EMA to every sample so it shows up as a W&B curve.
        samples = await super()._score_group(group)
        # Pop base rewards for this group's completions regardless of the
        # early-return below, so the dict's lifetime tracks the group and can't
        # leak when super() yields no samples.
        base_rewards = [self._rollout_base_rewards.pop(id(c), 0.0) for c in group.completions]
        statuses = [self._rollout_statuses.pop(id(c)) for c in group.completions]
        # Popped here (not below) so the dict can't leak when super() yields no
        # samples; None covers episodes that skipped the judge.
        judge_windows = [self._rollout_judge_window.pop(id(c), None) for c in group.completions]
        judged = [w for w in judge_windows if w is not None]
        # Wall-clock span of judging on this group's critical path: first judge
        # start to last judge finish, counting overlap across inflight slots once.
        group_judge_span_ms = (
            (max(e for _, e in judged) - min(s for s, _ in judged)) * 1000 if judged else 0.0
        )
        # Generation phase of sampling_batch_seconds: generation start -> group
        # queued for scoring, i.e. token decode + the judge (which runs inside
        # _generate_one). Excludes the scoring-queue wait and _score_group.
        generation_phase_s = group.queued_at - group.started_at
        if not samples:
            return samples
        await asyncio.to_thread(self._save_calibration, group, samples, statuses)
        group_reward = sum(s.metrics["reward"] for s in samples) / len(samples)
        self._reward_ema = (
            group_reward
            if self._reward_ema is None
            else _REWARD_EMA_ALPHA * group_reward + (1 - _REWARD_EMA_ALPHA) * self._reward_ema
        )
        for s, window in zip(samples, judge_windows, strict=True):
            s.metrics["reward_ema"] = self._reward_ema
            s.metrics["judge_time_ms"] = (window[1] - window[0]) * 1000 if window is not None else 0.0
            s.metrics["group_judge_span_ms"] = group_judge_span_ms
            s.metrics["generation_phase_s"] = generation_phase_s

        judge_pct = group_judge_span_ms / 1000 / generation_phase_s if generation_phase_s > 0 else 0.0
        logger.info(
            f"[timing] generation_phase={generation_phase_s:.1f}s, "
            f"judge_span={group_judge_span_ms / 1000:.1f}s ({judge_pct:.0%} of generation)"
        )

        # DAPO dynamic sampling: drop groups with zero reward std. When every
        # rollout in a group lands on the same reward, the group-normalized
        # advantage is 0 for all of them — no gradient, just batch noise. The
        # async generate loop runs ahead, so dropping here makes the trainer pull
        # the next informative group from the buffer: oversample-and-filter. EMA
        # and calibration are updated above first, so monitoring still sees the
        # dropped group's reward (only training skips it).
        #
        # Decide on the pre-penalty (base verifier) reward, popped here: the
        # soft-overlong length penalty can spread an all-failed group's rewards
        # (e.g. agent_error 0.0 -> -0.9/-0.6) into nonzero std, which would sneak
        # a group with no actual task signal past the std filter. Checking the
        # base reward drops it before the penalty rescues it. (base_rewards is
        # popped above, before the early-return, to avoid leaking the dict.)
        # Exception: an all-mastered group (every rollout solved, base all 1.0)
        # that the soft-overlong penalty has spread is kept — the length variance
        # across correct solutions is a weak but real "solve and be concise"
        # gradient. Failed/partial groups (base < 1.0) get only length noise from
        # the penalty, so they still drop.
        base_flat = statistics.pstdev(base_rewards) < 1e-8
        pen_std = statistics.pstdev([s.metrics["reward"] for s in samples])
        mastered_length_signal = base_flat and base_rewards[0] == 1.0 and pen_std >= 1e-8
        if self._dynamic_sampling and base_flat and not mastered_length_signal:
            self._dropped_groups += 1
            logger.info(
                "[dynamic-sampling] dropped zero-advantage group "
                f"(base_reward={base_rewards[0]:.3f}); total dropped={self._dropped_groups}"
            )
            # The expansion loop below normally consumes these; drop them here so a
            # dropped group doesn't leak per-completion turn/truncation markers.
            for completion in group.completions:
                self._rollout_turns.pop(id(completion), None)
                self._overlong_completions.discard(id(completion))
            return []
        if self._dynamic_sampling:
            for s in samples:
                s.metrics["dynamic_sampling/dropped_groups"] = self._dropped_groups

        # Sample splitting: replace each whole-rollout sample with one sample per
        # assistant turn, each carrying its windowed context (loss masked to the
        # turn) and sharing the rollout's episode-level advantage. super() returns
        # one sample per completion in group.completions order, so the zip aligns.
        expanded = []
        for completion, s, status in zip(group.completions, samples, statuses, strict=True):
            turns = self._rollout_turns.pop(id(completion), None)
            overlong = id(completion) in self._overlong_completions
            self._overlong_completions.discard(id(completion))
            if status not in _MODEL_OUTCOME_STATUSES:
                # Scoring/infra failures keep their reward in group normalization
                # above, but the rollout itself is not trainable policy behavior.
                self._infra_dropped += 1
                continue
            if self._overlong_filtering and overlong:
                # DAPO overlong filtering: mask the loss of a length-truncated
                # episode by dropping its training samples. Its reward stayed in the
                # group advantage normalization above; it just yields no gradient.
                self._overlong_dropped += 1
                continue
            if not turns:
                expanded.append(s)  # defensive; a rollout should always have turns
                continue
            for win_ids, turn_ids, turn_mask, turn_lp in turns:
                expanded.append(
                    RolloutSample(
                        prompt=s.prompt,
                        completion=s.completion,
                        input_ids=win_ids + turn_ids,
                        completion_mask=[0] * len(win_ids) + turn_mask,
                        old_log_probs=[0.0] * len(win_ids) + turn_lp,
                        advantage=s.advantage,
                        model_version=s.model_version,
                        metrics=dict(s.metrics),
                    )
                )
        if self._overlong_filtering:
            for s in expanded:
                s.metrics["overlong_filtering/dropped_samples"] = self._overlong_dropped
        for s in expanded:
            s.metrics["infra_filtering/dropped_samples"] = self._infra_dropped
        return expanded

    def _windowed_messages(self, prompt, completion):
        # Context for the next assistant turn: the prompt (system + initial user),
        # the pinned prefix through the list_tools exchange (its tool result carries
        # the discovered tool list), and the CONTEXT_WINDOW_TURNS most recent turns.
        # `completion` holds only complete (assistant, tool) pairs here, so indices
        # are even and completion[i+1] is completion[i]'s tool result.
        return windowed_messages(prompt, completion, CONTEXT_WINDOW_TURNS)

    async def _execute_tool_calls(self, tool_calls, tool_dict):
        # Async override of the base (sync) dispatcher: the AWM env tools are now
        # coroutines (real async client, no .sync() wrapper), so await them. This lets
        # the per-turn env round-trips overlap across inflight slots via the event loop
        # instead of blocking it. Mirrors AsyncRolloutWorker._execute_tool_calls otherwise.
        tool_messages, n_calls, n_failures = [], 0, 0
        for tool_call in tool_calls:
            n_calls += 1
            function = tool_call["function"]
            name = function["name"]
            if name not in tool_dict:
                n_failures += 1
                result = {
                    "error": f"Unknown tool '{name}'. The only tools you can call directly are "
                    f"{sorted(tool_dict)}."
                }
            else:
                try:
                    result = tool_dict[name](**function.get("arguments", {}))
                    if asyncio.iscoroutine(result):
                        result = await result
                except Exception as error:
                    n_failures += 1
                    result = {"error": str(error)}
            tool_messages.append({"role": "tool", "name": name, "content": str(result)})
        return tool_messages, n_calls, n_failures

    def _length_penalty(self, length):
        # DAPO Eq. (soft punish): 0 below (L_max - L_cache), a linear ramp to -1
        # across the cache zone, -1 at/above L_max. L_max is the per-turn generation
        # cap (self.max_tokens); L_cache is set from --soft-overlong-cache.
        l_max = self.max_tokens
        l_cache = self._soft_overlong_cache
        if length <= l_max - l_cache:
            return 0.0
        if length >= l_max:
            return -1.0
        return ((l_max - l_cache) - length) / l_cache

    async def _generate_one(self, prompt, tool_dict):
        # Reimplements AsyncRolloutWorker._generate_one's multi-turn loop so we can
        # early-terminate on a tool-call format violation (the base loop has no such
        # hook and would otherwise run every turn to completion). On the first
        # format error we keep the partial completion and force reward -1.0, matching
        # the AWM paper's step-level rule; otherwise we score normally via the judge.
        env = tool_dict["call_tool"].__self__
        completion, completion_ids, completion_logprobs, tool_mask = [], [], [], []
        tool_call_count = 0
        tool_failure_count = 0
        iteration_num = 0
        max_iterations = self.max_tool_calling_iterations
        max_turn_len = 0  # longest single turn, for soft overlong punishment
        self._rollout_turns[id(completion)] = []
        while True:
            # Rebuild the windowed context each turn (no monotonic accumulation): the
            # agent generates under the same truncated context the training sample
            # will use, so the vLLM old_log_probs match the trainer's input exactly.
            # apply_chat_template / parse_response are synchronous CPU-bound calls
            # (Jinja render + tokenize/detokenize) that run every turn for every
            # inflight rollout. On the single rollout event loop they block long
            # enough to stall the worker heartbeat, which trips check_health and
            # tears down all env sockets at once; offload them to a thread so the
            # loop keeps ticking.
            prompt_ids = await asyncio.to_thread(
                self.tokenizer.apply_chat_template,
                self._windowed_messages(prompt, completion),
                return_dict=False,
                add_generation_prompt=True,
                tools=self.tools or None,  # `or None`: Llama bug: it renders tool boilerplate for tools=[]
                chat_template=self.chat_template,
                **self.chat_template_kwargs,
            )
            turn_ids, turn_logprobs = await self._generate_one_turn(prompt_ids)
            assistant_message = await asyncio.to_thread(parse_response, self.tokenizer, turn_ids)
            completion.append(assistant_message)
            turn_mask = self._turn_mask(turn_ids)
            # Capture this turn as its own training sample (windowed context + turn).
            self._rollout_turns[id(completion)].append(
                (prompt_ids, turn_ids, turn_mask, turn_logprobs)
            )
            completion_ids.extend(turn_ids)
            completion_logprobs.extend(turn_logprobs)
            tool_mask.extend(turn_mask)
            max_turn_len = max(max_turn_len, len(turn_ids))
            if len(turn_ids) >= self.max_tokens:
                # The final turn hit the per-turn generation cap (max_completion_length),
                # i.e. the model was cut off mid-generation. DAPO overlong filtering
                # treats the truncated episode's reward as noise and masks its loss;
                # we mark it here and drop its samples in _score_group. The reward is
                # still scored below so it stays in the group's advantage normalization.
                self._overlong_completions.add(id(completion))
            tool_calls = assistant_message.get("tool_calls")
            if tool_calls is None or (max_iterations is not None and iteration_num >= max_iterations):
                # Normal termination: score the finished episode with the LLM judge.
                t_judge = time.monotonic()
                try:
                    reward, status = await env._score_rollout()
                    base_reward = reward
                    if self._soft_overlong_punishment:
                        # DAPO soft overlong punishment: add a length penalty (0 to
                        # -1) as the longest turn approaches the generation cap, so
                        # the model learns to finish before it gets truncated.
                        reward += self._length_penalty(max_turn_len)
                except RuntimeError:
                    # The worker loop is shutting down (event loop / env client closing)
                    # while this rollout is still in flight — typically at a worker
                    # stop/restart or phase boundary. Not a server error: the env is
                    # fine, the rollout just got caught in teardown and its reward is
                    # discarded anyway; swallow it so the worker isn't marked failed
                    # and check_health doesn't abort the whole run.
                    reward, status = 0.1, "rollout_error"
                    base_reward = 0.1
                judge_end = time.monotonic()
                self._rollout_rewards[id(completion)] = reward
                self._rollout_base_rewards[id(completion)] = base_reward
                self._rollout_statuses[id(completion)] = status
                self._rollout_judge_window[id(completion)] = (t_judge, judge_end)
                await asyncio.to_thread(self._save_trajectory, env, prompt, completion, reward, status)
                return completion, completion_ids, completion_logprobs, tool_mask, tool_call_count, tool_failure_count

            tool_messages, n_calls, n_failures = await self._execute_tool_calls(tool_calls, tool_dict)
            tool_call_count += n_calls
            tool_failure_count += n_failures
            completion.extend(tool_messages)
            suffix_ids = self._get_tool_suffix_ids(tool_messages)
            completion_ids.extend(suffix_ids)
            completion_logprobs.extend([0.0] * len(suffix_ids))
            tool_mask.extend([0] * len(suffix_ids))
            if env.format_violation:
                # Step-level format violation -> early-terminate with r_t = -1.0.
                # Keep the partial completion so the advantage applies to the tokens
                # generated up to the violation; skip the judge entirely.
                self._rollout_rewards[id(completion)] = -1.0
                self._rollout_base_rewards[id(completion)] = -1.0
                self._rollout_statuses[id(completion)] = "format_violation"
                await env._close_session()
                await asyncio.to_thread(self._save_trajectory, env, prompt, completion, -1.0, "format_violation")
                return completion, completion_ids, completion_logprobs, tool_mask, tool_call_count, tool_failure_count
            iteration_num += 1

    def _save_trajectory(self, env, prompt, completion, reward, status):
        if TRAJECTORY_FILE is None:
            return
        record = {
            "scenario": env.scenario,
            "task_idx": env.task_idx,
            "reward": reward,
            "status": status,
            "prompt": prompt,
            "completion": completion,
        }
        with open(TRAJECTORY_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def _save_calibration(self, group, samples, statuses):
        if CALIBRATION_FILE is None:
            return

        status_counts = Counter(statuses)
        rewards = [sample.metrics["reward"] for sample in samples]
        valid_rollouts = sum(status in _MODEL_OUTCOME_STATUSES for status in statuses)

        if valid_rollouts == 0:
            classification = "infrastructure_failure"
        elif valid_rollouts < len(statuses):
            classification = "uncertain"
        elif statistics.pstdev(rewards) > 0:
            classification = "learnable"
        elif all(status == "incomplete" for status in statuses):
            classification = "all failed"
        elif all(status == "complete" for status in statuses):
            classification = "mastered"
        else:
            classification = "model_misbehavior"

        record = {
            "scenario": group.reward_kwargs["scenario"][0],
            "task_idx": group.reward_kwargs["task_idx"][0],
            "task": group.prompt[-1]["content"],
            "model_version": group.model_version,
            "num_rollouts": len(rewards),
            "mean_reward": statistics.fmean(rewards),
            "reward_std": statistics.pstdev(rewards),
            "status_counts": dict(status_counts),
            "valid_rollouts": valid_rollouts,
            "classification": classification,
        }
        with open(CALIBRATION_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _verifier_reward(self, completions, **kwargs):
        rewards = []
        for c in completions:
            if id(c) not in self._rollout_rewards:
                logger.warning("verifier reward missing for completion id=%d; defaulting to 0.0", id(c))
            rewards.append(self._rollout_rewards.pop(id(c), 0.0))
        return rewards


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def build_dataset(
    env_url: str,
    dataset_size: int,
    dataset_start: int = 0,
    num_scenarios: int | None = None,
) -> Dataset:
    """List AWM scenarios/tasks and build the GRPO prompt dataset.

    When num_scenarios is set, the dataset is sized by unique-scenario coverage
    instead of dataset_size: scan the shuffled rows from dataset_start until that
    many distinct scenarios have been seen, and take exactly those rows.
    """
    env = AWMEnv(base_url=env_url).sync()
    with env:
        result = env.step(CallToolAction(tool_name="__list_scenarios__", arguments={}))
        scenarios = result.observation.scenarios

    prompts, scenario_names, task_indices = [], [], []
    for scenario in scenarios:
        for task_idx, task in enumerate(scenario["tasks"]):
            prompts.append(
                [
                    {"role": "system", "content": SYSTEM_PROMPT.format(today=date.today().isoformat())},
                    {"role": "user", "content": task},
                ]
            )
            scenario_names.append(scenario["name"])
            task_indices.append(task_idx)

    # Shuffle before truncating: the scenario list is ordered, so taking the
    # first dataset_size rows would both bias the dataset to early scenarios and
    # iterate one scenario at a time (making per-step reward track scenario
    # difficulty instead of training progress).
    dataset = Dataset.from_dict(
        {
            "prompt": prompts,
            "scenario": scenario_names,
            "task_idx": task_indices,
        }
    ).shuffle(seed=42)
    # The fixed seed makes the shuffled order identical across runs, so a
    # warm-started run can continue at index dataset_start (= sum of prior
    # runs' dataset sizes) instead of replaying the same rows.
    if num_scenarios is not None:
        # Grow the slice from dataset_start until num_scenarios distinct
        # scenarios are covered; the dataset size is whatever that took.
        seen: set[str] = set()
        end = dataset_start
        for name in dataset["scenario"][dataset_start:]:
            seen.add(name)
            end += 1
            if len(seen) >= num_scenarios:
                break
        if len(seen) < num_scenarios:
            # Coverage target unreachable from this start: the loop consumed the
            # rest of the dataset. Warn instead of silently training on everything.
            logger.warning(
                "Requested num_scenarios=%d but only %d unique scenarios are "
                "available from dataset_start=%d; using the full remaining %d groups.",
                num_scenarios, len(seen), dataset_start, end - dataset_start,
            )
    else:
        end = min(dataset_start + dataset_size, len(dataset))
    return dataset.select(range(dataset_start, end))


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------


def task_reward(completions, **kwargs):
    """Placeholder reward func required by AsyncGRPOTrainer's constructor.

    Actual scoring is done by AWMRolloutWorker._verifier_reward, which
    replaces this in self.reward_funcs after the worker is constructed.
    """
    return [0.0] * len(completions)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Async GRPO training for AWM agent tasks.")
    parser.add_argument(
        "--model-id",
        default=MODEL_CONFIG["id"],
        help="Base HF model, or a previous run's output dir / checkpoint-N dir / Hub"
        " repo to warm-start from (continual training). Use a fresh --output-dir.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        default=MODEL_CONFIG["resume_from_checkpoint"],
        help="Checkpoint-N directory to resume exactly, including optimizer, scheduler,"
        " RNG, and global step. Use a --max-steps value above the saved step.",
    )
    parser.add_argument("--env-url", default=ENVIRONMENT_CONFIG["url"])
    parser.add_argument("--output-dir", default=TRAINING_CONFIG["output_dir"])
    parser.add_argument("--dataset-size", type=int, default=DATASET_CONFIG["size"])
    parser.add_argument(
        "--dataset-start",
        type=int,
        default=DATASET_CONFIG["start"],
        help="Start index into the shuffled dataset; for continual training set"
        " this to the sum of previous runs' dataset sizes.",
    )
    parser.add_argument(
        "--num-scenarios",
        type=int,
        default=DATASET_CONFIG.get("num_scenarios"),
        help="If set, size the dataset by unique-scenario coverage instead of"
        " --dataset-size: scan the shuffled dataset from --dataset-start until"
        " this many distinct scenarios are covered. Overrides --dataset-size.",
    )
    parser.add_argument("--num-generations", type=int, default=ROLLOUT_CONFIG["num_generations"])
    parser.add_argument(
        "--dynamic-sampling",
        action="store_true",
        default=DAPO_CONFIG["dynamic_sampling"],
        help="DAPO dynamic sampling: drop groups whose rollouts all get the same"
        " reward (zero advantage / no gradient) and let the async generate loop"
        " refill with informative groups. Default from config (dapo.dynamic_sampling).",
    )
    parser.add_argument("--no-dynamic-sampling", dest="dynamic_sampling", action="store_false")
    parser.add_argument(
        "--overlong-filtering",
        action="store_true",
        default=DAPO_CONFIG["overlong_filtering"],
        help="DAPO overlong filtering: mask the loss of episodes whose generation"
        " was cut off at --max-completion-length (their reward is noise). The reward"
        " still counts in group normalization. Default from config (dapo.overlong_filtering).",
    )
    parser.add_argument("--no-overlong-filtering", dest="overlong_filtering", action="store_false")
    parser.add_argument(
        "--soft-overlong-punishment",
        action="store_true",
        default=DAPO_CONFIG["soft_overlong_punishment"],
        help="DAPO soft overlong punishment: add a length penalty (0 to -1) to the"
        " reward as the longest turn approaches --max-completion-length. Default from"
        " config (dapo.soft_overlong_punishment).",
    )
    parser.add_argument(
        "--no-soft-overlong-punishment", dest="soft_overlong_punishment", action="store_false"
    )
    parser.add_argument(
        "--soft-overlong-cache",
        type=int,
        default=DAPO_CONFIG["soft_overlong_cache"],
        help="L_cache token window for --soft-overlong-punishment: the penalty ramps"
        " from 0 to -1 over the last L_cache tokens before the cap. Defaults to 20%%"
        " of --max-completion-length (matching DAPO's 4096/20480).",
    )
    parser.add_argument("--max-turns", type=int, default=ROLLOUT_CONFIG["max_turns"])
    parser.add_argument(
        "--context-window-turns",
        type=int,
        default=ROLLOUT_CONFIG["context_window_turns"],
        help="Each per-turn training sample keeps system + initial user + the prefix"
        " through the list_tools exchange + this many most recent turns.",
    )
    parser.add_argument(
        "--max-completion-length",
        type=int,
        default=ROLLOUT_CONFIG["max_completion_length"],
    )
    parser.add_argument(
        "--thinking-token-budget",
        type=int,
        default=ROLLOUT_CONFIG["thinking_token_budget"],
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=TRAINING_CONFIG["gradient_accumulation_steps"],
    )
    parser.add_argument(
        "--per-device-batch-size",
        type=int,
        default=TRAINING_CONFIG["per_device_batch_size"],
    )
    parser.add_argument(
        "--learning-rate", type=float, default=TRAINING_CONFIG["learning_rate"]
    )
    parser.add_argument("--optim", default=TRAINING_CONFIG["optim"])
    parser.add_argument(
        "--max-steps",
        type=int,
        default=TRAINING_CONFIG["max_steps"],
        help="Absolute optimizer-step target; -1 derives it from dataset size and epochs.",
    )
    parser.add_argument("--reset-concurrency", type=int,
                        default=ENVIRONMENT_CONFIG["reset_concurrency"],
                        help="Max simultaneous env connect+reset calls.")
    parser.add_argument("--num-epochs", type=int, default=TRAINING_CONFIG["num_epochs"])
    parser.add_argument("--save-steps", type=int, default=CHECKPOINTING_CONFIG["save_steps"])
    parser.add_argument(
        "--save-total-limit", type=int, default=CHECKPOINTING_CONFIG["save_total_limit"]
    )
    parser.add_argument(
        "--logging-steps", type=int, default=CHECKPOINTING_CONFIG["logging_steps"]
    )
    parser.add_argument("--vllm-server-host", default=VLLM_CONFIG["host"])
    parser.add_argument("--vllm-server-port", type=int, default=VLLM_CONFIG["port"])
    parser.add_argument(
        "--vllm-server-timeout", type=float, default=VLLM_CONFIG["timeout_s"]
    )
    parser.add_argument(
        "--push-to-hub", action="store_true", default=HUB_CONFIG["push_to_hub"]
    )
    parser.add_argument("--no-push-to-hub", dest="push_to_hub", action="store_false")
    parser.add_argument("--wandb-project", default=WANDB_CONFIG["project"])
    parser.add_argument("--wandb-name", default=WANDB_CONFIG["name"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import wandb

    from transformers import AutoTokenizer
    from trl.chat_template_utils import qwen3_chat_template

    is_main_process = os.environ.get("LOCAL_RANK", "0") == "0"
    if is_main_process:
        wandb.login(key=os.environ.get("WANDB_API_KEY"))
        wandb.init(project=args.wandb_project, name=args.wandb_name)

    global TRAJECTORY_FILE, CALIBRATION_FILE, CONTEXT_WINDOW_TURNS, _RESET_SEMAPHORE
    _RESET_SEMAPHORE = asyncio.Semaphore(args.reset_concurrency)
    os.makedirs(args.output_dir, exist_ok=True)
    TRAJECTORY_FILE = os.path.join(args.output_dir, "rollouts.jsonl")
    CALIBRATION_FILE = os.path.join(args.output_dir, "calibration.jsonl")
    if args.resume_from_checkpoint is None:
        for f in (TRAJECTORY_FILE, CALIBRATION_FILE):
            if os.path.exists(f):
                os.remove(f)
    CONTEXT_WINDOW_TURNS = args.context_window_turns

    if args.resume_from_checkpoint is not None:
        import json as _json
        state_path = os.path.join(args.resume_from_checkpoint, "trainer_state.json")
        if os.path.exists(state_path):
            with open(state_path) as _f:
                _ckpt_state = _json.load(_f)
            _global_step = _ckpt_state.get("global_step", 0)
            _world_size = int(os.environ.get("WORLD_SIZE", 1))
            _rows_consumed = _global_step * args.per_device_batch_size * _world_size * args.gradient_accumulation_steps
            args.dataset_start += _rows_consumed
            if is_main_process:
                print(f"Resume: checkpoint step={_global_step}, rows_consumed={_rows_consumed}, "
                      f"dataset_start adjusted to {args.dataset_start}")

    dataset = build_dataset(
        args.env_url, args.dataset_size, args.dataset_start, args.num_scenarios
    )

    # When --num-scenarios is set it fully determines the run: build_dataset
    # already ignores --dataset-size, and here --max-steps is ignored too by
    # resetting it so AsyncGRPOTrainer re-derives the step target from
    # len(dataset). The trainer computes that with accelerator.num_processes,
    # the authoritative world size under FSDP2/multi-GPU — don't second-guess it
    # from WORLD_SIZE here (it may be unset, which would mis-scale max_steps).
    if args.num_scenarios is not None:
        args.max_steps = -1

    # Point the trainer at our subclass so it instantiates AWMRolloutWorker
    # instead of the base. The trainer still handles all weight-metadata and
    # tokenizer setup; we just swap the class before it calls AsyncRolloutWorker().
    from trl.experimental.async_grpo import async_grpo_trainer
    async_grpo_trainer.AsyncRolloutWorker = AWMRolloutWorker
    AWMRolloutWorker._dynamic_sampling = args.dynamic_sampling
    AWMRolloutWorker._overlong_filtering = args.overlong_filtering
    AWMRolloutWorker._soft_overlong_punishment = args.soft_overlong_punishment
    AWMRolloutWorker._soft_overlong_cache = (
        args.soft_overlong_cache
        if args.soft_overlong_cache is not None
        else args.max_completion_length // 5
    )

    # Qwen3-4B-Instruct-2507 ships its own chat template, which doesn't byte-for-byte
    # match any template TRL knows, so add_response_schema() inside AsyncRolloutWorker
    # would raise. Swap in TRL's bundled qwen3-instruct-2507 template (same <tool_call>
    # format) so the schema is recognized, and pass the tokenizer to the trainer.
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.chat_template = qwen3_chat_template

    grpo_config = AsyncGRPOConfig(
        model_init_kwargs={"attn_implementation": "flash_attention_3"},
        # Training schedule / optimization
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        # Rollout batches are generated asynchronously and cannot be replayed by
        # Transformers' normal dataloader fast-forward on checkpoint resume.
        ignore_data_skip=args.resume_from_checkpoint is not None,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        lr_scheduler_type="constant",
        optim=args.optim,
        # Pre-clip grad norms run 0-10 in practice; clipping at 1.0 was scaling
        # most updates down 5-10x. Clip only the rare outliers (~35).
        max_grad_norm=10.0,

        # GRPO configuration
        # Qwen3 thinking-mode recommended sampling temperature. Used both to
        # sample rollouts (sent in the vLLM request, overriding the server) and
        # to scale the training-loss logits, keeping the two consistent.
        # top_p/top_k/min_p/presence_penalty aren't exposed here — they're set on
        # the vLLM server in scripts/run_vllm_awm.sh.
        temperature=1,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        thinking_token_budget=args.thinking_token_budget,
        max_tool_calling_iterations=args.max_turns,
        # Importance sampling level (config dapo.importance_sampling_level):
        # "token" = raw per-token ratios (GRPO/DAPO); "sequence" = GSPO (one
        # length-normalized ratio per rollout); "sequence_token" = GSPO-token. The
        # AWM paper used GSPO; "token" matches the DAPO recipe (per-token ratios +
        # clip-higher + token-level loss).
        importance_sampling_level=DAPO_CONFIG["importance_sampling_level"],
        loss_type=DAPO_CONFIG["loss_type"],
        epsilon_high=0.28,  # DAPO-style high clip for more exploration
        # No KL penalty: the async trainer has no reference model, and old_log_probs
        # are vLLM sampling logprobs, not reference logprobs. GSPO with beta=0.
        log_completions=True,
        num_completions_to_print=2,
        # chat_template_kwargs={"enable_thinking": False},
        weight_sync_steps=1,
        max_staleness=ROLLOUT_CONFIG["max_staleness"],
        max_inflight_tasks=ROLLOUT_CONFIG["max_inflight_tasks"],
        # queue_maxsize=2304,
        heartbeat_stale_after_s=1200.0,
        # vLLM (async => server mode on a separate GPU)
        vllm_server_base_url=f"http://{args.vllm_server_host}:{args.vllm_server_port}",
        # How long the trainer waits on an empty rollout queue before stopping the
        # epoch (also the vLLM-server-ready timeout). Rollouts hit the AWM env
        # synchronously over multiple turns, so the default 240s can starve.
        vllm_server_timeout=args.vllm_server_timeout,

        # Precision
        bf16=True,

        # Logging / reporting
        output_dir=args.output_dir,
        report_to="wandb",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,

        # Memory
        gradient_checkpointing=True,

        # Hub
        push_to_hub=args.push_to_hub,
    )

    trainer = AsyncGRPOTrainer(
        model=args.model_id,
        reward_funcs=[task_reward],
        train_dataset=dataset,
        args=grpo_config,
        processing_class=tokenizer,
        environment_factory=lambda: AWMEnvironment(args.env_url),
    )

    # AsyncGRPOTrainer.__init__ derives max_steps from len(dataset) and
    # accelerator.num_processes when it's <= 0; surface the resulting step target.
    if trainer.is_world_process_zero():
        print(f"Training to max_steps={trainer.args.max_steps} over {len(dataset)} dataset groups")

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # No separate final checkpoint: when save_steps < max_steps the periodic
    # save_steps checkpoints already cover the run; when save_steps >= max_steps
    # no intermediate checkpoint is written at all. In both cases the Hub upload
    # captures the final weights, so a final checkpoint dir would just duplicate
    # it. push_to_hub saves the model first (distributed-safe under FSDP2: every
    # rank gathers the sharded state dict, only the main process uploads).
    if args.push_to_hub:
        trainer.push_to_hub(commit_message="Upload model")


if __name__ == "__main__":
    main()
