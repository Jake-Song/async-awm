---
name: inspect-rollouts
description: Use this skill when asked to inspect, summarize, or debug a rollouts.jsonl trajectory file from an async-GRPO training run (e.g. "look at the rollouts", "why are rewards low", "show me a failing trajectory").
---

Inspect a `rollouts.jsonl` file written by `open-env/openenv_awm_async_grpo.py` (`_save_trajectory` appends one JSON line per finished rollout, in completion order).

### Part 1: Locate the file

If the user didn't give a path, look for (in order): a path mentioned in the conversation, `<output_dir>/rollouts.jsonl` from the run being discussed, `experiment/rollouts.jsonl`, or `rollouts.jsonl` in the repo root (typically scp'd back from the GPU node). Files are tens of MB — never load with the Read tool; stream with python via Bash.

### Part 2: Know the schema

One JSON object per line:

- `scenario` (str), `task_idx` (int) — which AWM task.
- `reward` (float) — verifier reward. Older runs are mostly discrete: 1.0 = task done, 0.1 = incomplete/partial or scoring-failure floor, 0.0 = agent_error, −1.0 = invalid action. Newer graded/verifier or DAPO runs may contain continuous rewards and negative length/truncation penalties, so do not assume only 1.0/0.1/0.0.
- `status` (str) — judged outcomes: `complete`, `incomplete`, `agent_error`, plus scoring/infrastructure failures such as `judge_error` / `llm_judge_error` / `no_verifier` / `server_error` / `code_verify_error`, or `env_error:<ExceptionType>` for client-side errors talking to the env. `thread_error` / `rollout_error` are usually caught in worker teardown/shutdown paths. `format_violation` = invalid action.
- `prompt` — list of `{role, content}` (system + user task).
- `completion` — multi-turn message list: assistant messages (may carry `tool_calls`) interleaved with `tool` messages (`name` + `content` = tool result).

Line order ≈ training order, so reward over line index shows training progress (groups of `--num-generations`, default 8, share a prompt).

### Part 3: Always start with an overview

```bash
python3 - <<'EOF'
import json, collections
path = "rollouts.jsonl"
n, rew, status, scen = 0, [], collections.Counter(), collections.defaultdict(list)
for line in open(path):
    r = json.loads(line); n += 1
    rew.append(r["reward"]); status[r["status"]] += 1
    scen[r["scenario"]].append(r["reward"])
print(f"{n} rollouts, mean reward {sum(rew)/n:.3f}, statuses {dict(status)}")
k = max(1, n // 10)
print("reward by tenth:", [round(sum(rew[i:i+k])/len(rew[i:i+k]), 2) for i in range(0, n, k)])
for s, rs in sorted(scen.items(), key=lambda kv: sum(kv[1])/len(kv[1])):
    print(f"{s:24s} n={len(rs):4d} mean={sum(rs)/len(rs):.3f}")
EOF
```

Also report the exact reward-value histogram (`collections.Counter` over rewards) — the 0.0 vs 0.1 split distinguishes "failed" from "partial" — and the **zero-variance group count**: group rewards by `(scenario, task_idx)` and count groups where all rewards are identical. Those groups have zero GRPO advantage and contribute no gradient; if a large share is zero-variance, say so — it explains a flat reward curve better than any per-trajectory pathology, and points to difficulty filtering as the fix.

**Count zero-variance over FULL groups only** (size `--num-generations`, typically 8, or a multiple like 16 when a task is resampled across steps). Groups whose size isn't a clean multiple are force-ended tail partials from run shutdown — they read as zero-variance on too few samples and inflate the count. Report both numbers but lead with the full-group fraction. Example pitfall: an all-groups count of 20/57 (35%) dropped to 12/43 (28%) once the 14 truncated partials were excluded.

### Part 3b: Check learning signal and difficulty calibration

Zero-variance counts *strictly identical* rewards; calibration is the fuller picture. Treat reward as solve/fail (1.0 = solved, else not) and compute each full group's **solve-rate**, then bucket. Lead with the strict learnable percentage: a full group is **clearly learnable** only when the same `(scenario, task_idx)` has at least one solved rollout (`reward == 1.0`) and at least one not-solved rollout (`reward != 1.0`). That is the cleanest GRPO signal because the model can contrast successful and failed trajectories for the same prompt.

```bash
python3 - <<'EOF'
import json, collections, os, statistics
g=collections.defaultdict(list)
for l in open("rollouts.jsonl"):
    r=json.loads(l); g[(r["scenario"],r["task_idx"])].append(r["reward"])
full={k:v for k,v in g.items() if len(v)>=8 and len(v)%8==0}  # adjust if num-generations is not 8
rates=[sum(x==1.0 for x in v)/len(v) for v in full.values()]
b=collections.Counter()
for s in rates:
    b["never (0%)" if s==0 else "always (100%)" if s==1 else
      "(0,25%)" if s<.25 else "[25,50%)" if s<.5 else "[50,75%)" if s<.75 else "[75,100%)"]+=1
full_rollouts=sum(len(v) for v in full.values())
strict=sum(0<s<1 for s in rates)
reward_spread=sum(1 for v in full.values() if len(set(v))>1)
zero_var=sum(1 for v in full.values() if len(set(v))==1)
print(f"groups: all={len(g)} full={len(full)} partial/non-multiple={len(g)-len(full)}")
print(f"strict learnable solve-spread={strict}/{len(full)} ({strict/len(full)*100:.1f}% of full groups; {sum(len(v) for v,s in zip(full.values(),rates) if 0<s<1)/full_rollouts*100:.1f}% of full-group rollouts)")
print(f"any reward-spread={reward_spread}/{len(full)} ({reward_spread/len(full)*100:.1f}%)")
print(f"zero-variance={zero_var}/{len(full)} ({zero_var/len(full)*100:.1f}%)")
for k in ["never (0%)","(0,25%)","[25,50%)","[50,75%)","[75,100%)","always (100%)"]:
    print(f"  {k:14s} {b[k]:3d}  {b[k]/len(full)*100:4.0f}%")
print(f"mean solve-rate={statistics.mean(rates):.2f} (GRPO-ideal ~0.5)")
if os.path.exists("calibration.jsonl"):
    c=collections.Counter(json.loads(l).get("classification") for l in open("calibration.jsonl"))
    total=sum(c.values())
    if total:
        print(f"calibration classifications={dict(c)}")
        if "learnable" in c:
            print(f"calibration learnable={c['learnable']}/{total} ({c['learnable']/total*100:.1f}%)")
EOF
```

Report: mean solve-rate (near 0.5 = well-calibrated), strict learnable percentage, any reward-spread percentage, zero-variance percentage, and the **never-vs-always asymmetry** — more never-solved than always-solved means dead-hard tasks are the bigger drag, so filtering should weight toward pruning/replacing those over the trivial ones. Keep the definitions separate: "contains complete samples" is not enough; the useful strict signal is complete and non-complete samples inside the same full group.

Then drill into whatever the user actually asked about.

### Part 4: Read individual trajectories

To answer "why" questions (low reward, weird behavior), print a few matching trajectories. Filter on scenario/status/reward, and truncate tool outputs so the transcript stays readable:

```bash
python3 - <<'EOF'
import json, itertools
def show(r):
    print(f"=== {r['scenario']}#{r['task_idx']} reward={r['reward']} status={r['status']}")
    print("TASK:", r["prompt"][-1]["content"][:300])
    for m in r["completion"]:
        if m["role"] == "assistant":
            for tc in m.get("tool_calls") or []:
                f = tc["function"]; print(f"  -> {f['name']}({json.dumps(f['arguments'])[:200]})")
            if m["content"]: print("  ASSISTANT:", m["content"][:300])
        else:
            print(f"  <- {m.get('name')}:", str(m["content"])[:200])
rows = (json.loads(l) for l in open("rollouts.jsonl"))
match = (r for r in rows if r["status"] == "incomplete")  # adjust filter
for r in itertools.islice(match, 3): show(r)
EOF
```

Useful variants: compare a `complete` vs an `incomplete` rollout of the same `(scenario, task_idx)`; count tool calls per rollout (`sum(len(m.get("tool_calls") or []) for m in r["completion"] if m["role"]=="assistant")`); grep tool results for error strings.

### Part 4b: Triage failures into known categories

Classify each failing rollout by its tool-call names — this splits the failure mass into actionable buckets:

- **`no_tool_calls`** — zero tool calls: the model refused from chat habit ("I don't have access to..."), common on finance/payment tasks. Fix lives in the system prompt.
- **`direct_mcp_names`** — any call whose name is not `list_tools`/`call_tool`: the model called an MCP tool by name instead of through the `call_tool` wrapper. Signature: tool results like `{'error': "'list_collections'"}` (a bare `KeyError` repr from `tool_dict[name]`). Count these wasted turns; models often conclude "the system is broken" and give up.
- **`proper_wrapper`** — mechanics correct, content wrong. Look for grounding errors: the model inventing absolute dates for relative ones ("next 14 days" → queries from `2025-01-01`), then trusting an empty result. Check whether the system prompt includes today's date.

Also check the truncation signature. For thinking models, do not rely only on "empty last message"; many harmful truncations are huge final assistant messages with an open `<think>` and no closing `</think>`. Split the count into **harmful** and **harmless/already-complete**:

```bash
python3 - <<'EOF'
import json
rows=[json.loads(l) for l in open("rollouts.jsonl")]
def last_assistant(r):
    for m in reversed(r.get("completion", [])):
        if m.get("role")=="assistant": return m
    return {}
def text(m):
    return ((m.get("content") or "") + "\n" + (m.get("reasoning_content") or "")).strip()
def open_think(m):
    t=text(m); return t.count("<think>") > t.count("</think>")
n=len(rows)
open_rows=[r for r in rows if open_think(last_assistant(r))]
harmful=[r for r in open_rows if r.get("reward",0) < 0]
harmless=[r for r in open_rows if r.get("reward",0) >= 0]
negative=[r for r in rows if r.get("reward",0) < 0]
print(f"open final <think>={len(open_rows)}/{n} ({len(open_rows)/n*100:.1f}%)")
print(f"harmful open-think negative={len(harmful)}/{n} ({len(harmful)/n*100:.1f}%)")
print(f"harmless/already-complete open-think={len(harmless)}/{n} ({len(harmless)/n*100:.1f}%)")
if negative:
    print(f"open-think share of negative rewards={len(harmful)}/{len(negative)} ({len(harmful)/len(negative)*100:.1f}%)")
EOF
```

Quote the harmful percentage as the reward-damaging truncation rate. Mention the broader open-`<think>` percentage only as a symptom rate, because some open-think rollouts are still scored `complete` after the environment state was already correct.

### Part 5: Report to LOG.md

Summarize findings in prose: overall reward + trend, which scenarios drag the mean down, and what the failing trajectories actually do (e.g. "model claims it has no tools instead of calling list_tools"). Quote short transcript excerpts as evidence.

Append the report to `experiment/LOG.md` (the canonical training log; create it if missing) under a heading like `## YYYY-MM-DD — <path to rollouts.jsonl>`, then give the user a brief recap in chat and point them to the LOG.md entry.
