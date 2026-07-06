# Analysis of `rollouts.jsonl` (2026-06-11)

1,446 rollouts, 163 scenarios, 180 GRPO groups. Branch: `relax-format-violation`.

**Headline finding: a large share of failures trace to one mechanical bug — the model calls env tools directly by name instead of through the `call_tool` wrapper, and the env replies with a useless raw `KeyError` string.**

## Reward overview

- Pass rate **38.8%** (561 of 1,446 at reward 1.0). The rest: 610 at 0.1, 275 at 0.0 — both marked `incomplete`; the 0.0-vs-0.1 split comes from the AWM env verifier itself, not the trainer.
- No upward trend across the file: per-decile mean reward bounces between 0.32 and 0.64 with no slope, so the run isn't visibly learning yet.

## The `call_tool` bypass bug

The system prompt tells the model to use `list_tools` then `call_tool`, but in **521 rollouts (38% of tool-using rollouts)** the model emits the discovered tool name directly (e.g. `list_forms(project_id=1)`). There were 1,684 such direct calls and **99.7% of them fail**, returning `{'error': "'list_forms'"}` — a stringified Python `KeyError` that gives the model no hint it should use `call_tool` instead. Models that did this pass at 33.6% vs 43.8% for wrapper-only rollouts. A typical trajectory: direct call fails → retries the same call → concludes "the environment has a limitation" → gives up or hallucinates success.

Two cheap fixes, either of which should lift reward: have the dispatcher return `Unknown tool 'X' — invoke it via call_tool(name='X', args={...})`, or simply register the discovered tools as directly callable.

## `call_tool` error breakdown (500 errors out of 2,608 calls, 19.2%)

- **354 HTTP 500 Internal Server Error** — env-side, not model fault (consistent with the recent 0.1-for-server-error commits, but note these surface inside tool responses mid-rollout, so the rollout still proceeds and typically ends 0.0/0.1).
- **126 argument validation errors** (e.g. `None is not of type 'integer'`, `int_parsing`) — genuine model mistakes, recoverable.
- 20 not-found/404.

## Failure modes among the 885 failed rollouts

- **478 gave up citing errors/inability** — heavily driven by the KeyError bug and the 500s above; 288 failed rollouts contain 3+ error tool responses (flailing on repeats).
- **244 claimed success but the verifier disagreed** — e.g. "I have successfully created a new Calendar view..." at reward 0.1; either genuine near-misses or false completion claims.
- **52 never called a single tool** — hallucinated "I don't have access to Zelle/banking tools" and refused (all of `banking_4` failed this way, 16/16 at 0.0).
- 111 other.

## GRPO signal efficiency

- **55% of groups (99/180) have zero reward variance** — 41 all-pass, 58 all-fail — contributing zero gradient. Only 81 groups carry signal.
- Of the mixed groups, 28 are (0.0, 0.1)-only mixtures: the gradient there only pushes between two failure modes, which is weak signal at best. Truly informative pass/fail contrast exists in ~53 groups (~29%).

## Bottom line

Before tuning reward shaping further, fixing the `call_tool` KeyError response is the single highest-leverage change — it's implicated in roughly a third of all rollouts and directly suppresses pass rate, and the 354 mid-rollout 500s are worth an env-side look too.

---

# Status analysis of `rollouts.jsonl` (2026-06-11, 3,263-rollout snapshot)

Newer, larger snapshot (3,263 rollouts). Branch: `relax-format-violation`.

## Status distribution

| Status | Count | % |
|---|---|---|
| incomplete | 1,652 | 50.6% |
| complete | 1,265 | 38.8% |
| format_violation | 278 | 8.5% |
| server_error | 68 | 2.1% |

## format_violation is scenario-concentrated

Top offenders: `hr_payroll_management_1` 24/44 (54.5%), `workforce_management_1` 36/80 (45.0%), `payments_billing_1` 20/80 (25.0%), `booking_and_scheduling_1` 19/80 (23.8%), `tournament_management_1` 17/80 (21.2%). The top 5 scenarios hold 116 of 278 violations (~42%); 8 scenarios have zero. `hr_payroll_management_1` is also short on rollouts (44 vs the usual 80), as are `survey_and_forms_1` (68) and `b2b_marketplace_2` (78) — truncation may correlate with whatever triggers the violations.

## server_error status undercounts real server errors

- The `server_error` *status* hits only 3 scenarios, all booking: `booking_and_appointments_1` 33/113 (29.2%), `booking_marketplace_1` 28/160 (17.5%), `booking_1` 7/80 (8.8%). The first two have inflated totals (113 and 160 vs the usual 80), i.e. they were resampled/retried.
- 7 of the 68 server_error rollouts still carry reward 0.0 despite the 0.1-floor commit — either pre-fix data or a bypassing path.
- Meanwhile, HTTP 500s appear *inside* rollouts classified as everything else: 220/1,652 incomplete (13.3%) contain a `Status code: 500` tool response, and **112 of those ended on the 500** — almost certainly killed by the server but labeled `incomplete` and denied the 0.1 floor. 125/1,265 complete (9.9%) and 68/278 format_violation (24.5%) also contain 500s.
- format_violation containing 500s at 2x the base rate suggests some "violations" are the model reacting badly after the env broke (workforce_management_1 / tournament_management_1 appear in both top lists).

## Bottom line

The status classifier only flags a server error when the episode-level call fails; mid-rollout 500s — including ones that terminate the trajectory — fall through to `incomplete`/`format_violation` and get punished as model failures. Classifying "last tool response was a 500" as server_error would roughly triple the bucket (68 → ~180) and remove that much noise from the reward signal.

---

# Next experiment: small-pool learning-signal probe (2026-06-11)

Hypothesis to test: shrinking the task pool makes any learning signal directly measurable — the RL equivalent of "overfit a small batch". If the policy can't improve on a handful of tasks it revisits many times, there is no usable learning signal at all, isolated from the measurement noise above.

## Why it works mechanically

At 2 groups/step, a 24-step run processes ~48 groups. The current ~1000-row dataset visits each task at most 1–2 times, which is why per-task curves were impossible. With a pool of ~15–20 tasks, every task gets revisited 2–3 times *within one run* — per-task reward becomes a real curve, and the paired visit-over-visit comparison (which already hinted at +0.11) gets statistical teeth.

## Setup requirements

1. **Loop the dataset.** With `num_train_epochs=1`, a 20-task dataset ends the run after ~10 steps. Raise epochs (or confirm the worker's repeat-iterator governs run length) so step count, not epoch count, decides when it stops.
2. **Pick tasks where signal can exist.** A group of 8 only produces gradient when outcomes differ (reward_std > 0). Known-learnable from the data: `booking_and_appointments_1` (0.1 → 1.0 on second visit), `marketplace_1` and `hr_payroll_management_1` task 2 (mixed outcomes within groups). Including 2–3 of the former format-trap tasks (payroll tasks 5/9) directly tests the `invalid_args` relaxation: deterministic -1.0 before, should retry past schema errors now.
3. **Select scenarios explicitly, not via `dataset_size`.** Truncating the shuffled dataset gives a random 20 tasks; a controlled probe needs a fixed hand-picked set — add a small scenario-name filter in `build_dataset`.

## Expected readout

- Per-task curves trending up on the learnable set → pipeline confirmed, scale back to the full dataset.
- Flat per-task curves despite repeated visits, healthy reward_std, and no format traps → genuine learning-signal problem; suspects become LR=7e-7 too small, advantage quality, or credit assignment over multi-turn tool masks.

## 2026-06-14 — rollouts.jsonl (repo root, 422 rollouts)

**Overview.** 422 rollouts, mean reward **0.354**. Reward histogram: `1.0`×124 (complete), `0.1`×255, `0.0`×43 (agent_error). No `-1.0` format violations. Status mix: complete 124, incomplete 123, agent_error 43, plus a large **non-model / scoring-failure band of 132 (31%)**: rollout_error 42, episode_already_done 35, server_error 24, no_verifier 16, judge_error 15 — all forced to 0.1.

**The end-of-run reward "collapse" is an infra artifact, not regression.** Reward by tenth: `0.40 0.43 0.43 0.32 0.41 0.26 0.46 0.42 0.32 0.12 0.10`. The last two tenths crater because the final 44 rollouts are **42× `rollout_error`** (all reward 0.1) — the env/trainer stack died at the tail. Drop that band and the curve is flat-ish around 0.40, no learning trend.

**Zero-variance groups: 11/54 (20%).** Not the bottleneck — most groups still produce GRPO advantage, so difficulty filtering isn't the priority here.

**Root cause of the model failures: the `call_tool` wrapper is not understood.** Real MCP tools must be invoked via `call_tool(name=..., arguments=...)`; only `call_tool`/`close_session`/`list_tools` are directly callable. The model repeatedly fails this and gives up. Triage of the 166 agent_error+incomplete rollouts: `proper_wrapper` 69, `direct_mcp_names` 61, `no_tool_calls` 36.

- **direct_mcp_names (61).** Model calls the MCP tool by name. Signature, `notes_knowledge_management_1#5`: after `list_tools` it calls `create_database` directly → `{'error': "Unknown tool 'create_database'. The only tools you can call directly are ['call_tool','close_session','list_tools']"}`. Wasted turns, then surrender.
- **no_tool_calls / giving up (36).** Model reads the tool list, sees only the 3 wrapper tools as directly callable, and concludes the task is impossible. `it_asset_management_1#8`: *"The tools available (call_tool, close_session, list_tools) do not support direct manipulation of device assignments... No function exists..."* — then stops or calls `close_session` (→ the 35 `episode_already_done`).

**Scope of the give-up pathology: 92/422 (22%)** of rollouts end on an explicit "not available / cannot be completed / does not exist" message. This single misunderstanding bleeds across buckets — `episode_already_done` (premature close_session), `direct_mcp_names`, `no_tool_calls`, and many `incomplete`.

**Truncation (`max_tokens`) is minor: 35/422** end with an empty final assistant message. Not the dominant problem.

**Worst scenarios** (mean ~0.05–0.10, near-total failure): `membership_management_4` 0.05, `it_asset_management_1` 0.075, `hr_system_1` 0.075, `healthcare_patient_portal_5` 0.083, plus a long tail of finance/booking/CRM tasks at 0.088–0.10. **Best:** `clinic_management_2` 1.00, `team_collaboration_1`/`gaming_2` 0.89, `content_bookmark_management_1` 0.875.

**Recommended fixes, in order.**
1. Fix the system prompt / tool docs so the model uses `call_tool(name, arguments)` from `list_tools` output instead of calling tool names directly or declaring the task impossible. This is the single highest-leverage change (~22% of rollouts surrender on it).
2. Investigate the tail `rollout_error` burst (last 44 rollouts) — env server / trainer stability, not the model.
3. Reduce the scoring-failure band (server_error/judge_error/no_verifier = 55 rollouts forced to 0.1) so reward reflects the policy.

## 2026-06-14 (21:51) — rollouts.jsonl (post system-prompt + Qwen3-sampling fix)

Follow-up to the 18:05 run, after (a) rewriting `SYSTEM_PROMPT` + the `list_tools`/
`call_tool` docstrings to hammer the wrapper pattern and forbid giving up, and
(b) Qwen3 thinking-mode sampling (temp 0.6 / top_p 0.95 / top_k 20 / min_p 0 /
presence_penalty 1.5 server-side, temp 0.6 trainer-side).

**Big win. Mean reward 0.354 → 0.547** (473 rollouts). Histogram: `1.0`×236 (was
124), `0.1`×228, `0.0`×9 (was 43). The give-up pathology is essentially solved:

| metric | 18:05 | 21:51 |
|---|---|---|
| mean reward | 0.354 | **0.547** |
| complete | 124 | **236** |
| agent_error (0.0) | 43 | **9** |
| "unavailable/cannot complete" endings | 92 (22%) | **9 (2%)** |
| no_tool_calls refusals | 36 | **9** |
| episode_already_done (premature close_session) | 35 | **0** |
| rollout_error (tail infra) | 42 | **5** |

The system-prompt/docstring changes did exactly what they targeted: the model no
longer declares tasks impossible or closes the session to bail.

**The trailing "0.1" in reward-by-tenth is an artifact** — it's a 3-row remainder
bucket, not a collapse. The last full tenth is 0.73; the tail rollout_errors are
down to 5.

**Remaining work — failures are now content errors, not mechanics.** Of 178
agent_error+incomplete: `proper_wrapper` **110** (correct call mechanics, wrong
content — grounding/logic), `direct_mcp_names` **59**, `no_tool_calls` 9.
- `direct_mcp_names` (59) persists: the model still intermittently emits a tool
  call named after a domain tool (e.g. `get_form_by_title`, `list_providers`)
  instead of wrapping it, usually mixing correct `call_tool` calls in the same
  rollout — a slip, not a misunderstanding. Lower-priority than before but not gone.
- `proper_wrapper` (110) is the new frontier: mechanics right, outcome wrong.
  Needs trajectory-level inspection of grounding (dates/ids/filters) rather than
  prompt tweaks.

**Minor:** truncation (empty final assistant) ticked up 35→47 (8%→10%) — worth
watching given presence_penalty=1.5 may lengthen/diverge generations. Scoring-
failure band (server_error 23, judge_error 15, no_verifier 16) roughly unchanged.

**Zero-variance groups 26/63 (41%, up from 20%)** — but now 10 are all-1.0
(mastered) vs 16 all-0.1, i.e. driven by tasks the model now solves perfectly, not
by uniform failure. Worst scenarios still ~0.09–0.10 (booking_platform_4,
iot_smart_infrastructure_management_1, banking_4, survey_forms_platform_1,
form_builder_1, practice_management_4); 6 scenarios now at mean 1.000.

### Drill-down — the 110 proper_wrapper content failures (21:51 run)

Mechanics are correct (only `list_tools`/`call_tool`); the outcome is wrong. They
split into three buckets:

**1. Premature completion / under-acting — the dominant failure (~72).** Median 2
tool calls; **87/110 make ≤1 `call_tool`**. The model does the single most
obvious mutation that name-matches the task, gets a success response, and declares
done — **63/110 final messages explicitly claim success** ("has been successfully
updated… The task is complete"), and of the 59 single-call cases, 39 claim success.
The verifier wants the *full* multi-part outcome. Examples:
- `billing_and_invoicing_2#7`: task = configure a dunning *sequence* (retry 3× at
  2-day intervals, set `past_due` after first failure, `canceled` after all retries,
  triggered by a `payment_failed` webhook). Model calls `create_dunning_policy`
  once and stops — never wires the webhook trigger or retry schedule.
- `form_builder_1#4`: task = change the submission status `in_review`→`approved`
  AND record an approval_step. Model patches the approval_step_instance only,
  misses the submission-status change, and also takes `id:1` as "oldest" without
  ordering. Wrong/partial entity.

**2. Trust-an-empty-result (38/110, overlaps #1).** The model calls one convenience
query tool, gets `[]`/null, and concludes "no data" instead of computing it from
the underlying tools. `messaging_communications_1#9`: calls
`get_team_performance_last_30_days` → `{"teams": []}` and gives up on the whole
per-team report rather than aggregating from conversations.

**3. Runaway thinking / truncated before acting (~20).** 28 PW failures make **zero**
`call_tool`; **20 of those end inside an unclosed `<think>` block** — the model
spends the full thinking budget reasoning ("First, I need to find the form…") and
is cut off before emitting any tool call. This is the `thinking_token_budget`
(1280) / `max_completion_length` ceiling biting on the planning-heavy tasks, and
likely the source of the truncation tick-up; presence_penalty=1.5 may lengthen
these. These are infra/budget, not reasoning, failures.

**Takeaways.** Buckets 1+2 (~90%) are a single behavior: the model stops too early
and over-claims success. This won't move with sampling — candidate fixes: (a) a
system-prompt rule to decompose multi-part tasks and verify each required mutation
before stopping (and never trust an empty query result — fall back to lower-level
tools); (b) consider a reward/verifier signal that exposes partial completion so
GRPO can push toward finishing. Bucket 3 (~20) argues for a higher
`thinking_token_budget` (or larger `max_completion_length`) on planning-heavy
scenarios. Whole-group failures (form_builder_1, messaging_communications_1,
practice_management_7, dating_2, billing_and_invoicing_2 all 8/8) are the place to
validate any fix.

### Zero-variance (std=0) group inspection (21:51 run)

26/63 groups have identical rewards across all 8 generations → **zero GRPO
advantage, no gradient (~41% of compute produces no learning signal)**. They split:

**all-1.0 — mastered, too easy (10 groups):** bookmark_manager_2#4,
content_bookmark_management_1#1, e_commerce_10#2, e_commerce_marketplace_9#0,
job_search_and_recruiting_1#3, peer_to_peer_payments_1#7, professional_networking_9#7,
social_media_4#5, task_management_5#7, team_collaboration_1#8.

**all-0.1 (16 groups) — two distinct causes, must NOT be lumped together:**

- **Scoring/infra dead tasks (~6, plus 2 mixed) — the model is NOT at fault.**
  banking_4#2 (`no_verifier`×8), tutoring_education_marketplace_1#2
  (`no_verifier`×8), practice_management_4#3 (`judge_error`×8),
  tournament_management_1#0 (`server_error`×8), marketplace_8#2 &
  workflow_automation_3#7 (`rollout_error`). Inspection shows the model often
  **completes these correctly** — banking_4#2 sends the Zelle transfer and reports
  success; practice/tutoring update the records and report success — but the task
  has no registered verifier / the judge errors / the env step crashes, so reward
  is force-set to 0.1 regardless. These can NEVER produce signal and are scored
  unfairly. They also drag their scenarios into the "worst" list artificially
  (banking_4, tutoring_education_marketplace_1 look like ~0.1 failures but aren't).
  tournament_management_1#0 is genuinely broken: `create_game` itself returns an
  internal server error.

- **Genuine uniform model failure (~8) — too hard, all 8 truly incomplete.**
  billing_and_invoicing_2#7, dating_2#3, form_builder_1#4,
  membership_management_4#2, messaging_communications_1#9, practice_management_7#1,
  stock_trading_3#5, ticketing_and_access_management_1#3. These are the multi-part
  / premature-stop content failures from the proper_wrapper drill-down.

**Implications.**
1. **Drop the scoring/infra dead tasks from the training set** (or fix their
   verifiers). They consume a full group's compute, give zero gradient, and
   penalize correct behavior — the worst category.
2. **Difficulty-filter** the all-1.0 (too easy) and uniform-fail (too hard) groups
   so generations concentrate on tasks with reward variance. ~41% zero-variance is
   the biggest single drag on training efficiency right now — a bigger lever than
   any per-trajectory prompt tweak.
3. The "worst scenarios" ranking is contaminated by unverifiable tasks; recompute
   scenario means after excluding `no_verifier`/`judge_error`/`server_error` before
   drawing difficulty conclusions.

## 2026-06-15 (19:35) — rollouts.jsonl (repo root, 814 rollouts)

**Best run so far: mean reward 0.549** (0.430 → 0.549), **0.631 excluding the
scoring-failure band**. Histogram: `1.0`×408, `0.1`×391, `0.0`×15. Statuses:
complete 408, incomplete 266, agent_error 15, plus a **scoring-failure band of 125
(15%)**: judge_error 38, server_error 38, rollout_error 33, no_verifier 16.
**The refusal pathology is solved — `no_tool_calls` 15 → 1.**

**The apparent end-of-run decline is an infra artifact, not regression.** Reward by
tenth: `0.59 0.45 0.49 0.64 0.42 0.80 0.52 0.69 0.52 0.38 0.33`. The last fifth
carries **33 `rollout_error` + the bulk of the scoring failures** (tenth 9 alone:
36/81 scoring failures), forcing those to 0.1. The run did **not** hard-crash — the
final rows are ordinary `incomplete`s. Strip the scoring band and the curve is
healthy around 0.6.

**Zero-variance groups 40/104 (38%, down from 51%):** 15 all-1.0 (mastered) +
25 all-0.1. Improving, but still ~38% of compute yields no GRPO gradient.

**`direct_mcp_names` recovery is now clean (improvement over the 16:51 run).**
Triage of the 266 incompletes: `proper_wrapper` 146, `direct_mcp_names` 119,
`no_tool_calls` 1; truncated 41. `direct_mcp_names` is still high *and bursty*
(per-tenth share swings 5/30 → 27/40), but the failure mode changed: the model
calls an MCP tool by name, gets `{'error': "Unknown tool '...'. The only tools you
can call directly are ['call_tool','list_tools']."}`, and **self-corrects with a
valid structured `call_tool`** — no more malformed `<tool_call>` text / broken JSON.
So these are now a wasted-turn slip, not a fatal misunderstanding, and the resulting
0.1s are usually *content* failures, not mechanics. Example, `survey_forms_platform_1#9`:
slips on `get_form_by_title`, recovers, calls `seed_synthetic_responses` → returns
`{"created_responses": 50, "aggregate_counts": {}}` — seeded the rows but the
`distribution_config` never applied, so the verifier marks it incomplete.

**`proper_wrapper` (146, 55%) is the real frontier** — same premature-stop /
wrong-content pattern as before (mechanics right, outcome partial). These plus the
content-failure direct_mcp slips are where the reward ceiling now sits.

**Worst scenarios (all 8/8 fail, mean ≤0.10):** b2b_workflow_line_of_business_app_1,
practice_management_7, stock_trading_3, task_management_16, banking_4,
survey_forms_platform_1, form_builder_1, practice_management_4,
messaging_communications_1, tutoring_education_marketplace_1, crm_6. Several are
repeat offenders / known scoring-failure dead tasks (banking_4, practice_management_4,
tutoring_education_marketplace_1). **Mastered (1.000):** team_collaboration_1,
clinic_management_2, live_streaming_1, peer_to_peer_payments_1, compliance_management_1,
task_management_5.

**Recommended, in order.**
1. **Stabilize infra — the scoring-failure band rose to 15%** (judge_error 38,
   server_error 38, rollout_error 33). The rollout_error tail burst is what makes the
   run *look* like it's regressing; fixing env/judge stability both cleans the signal
   and removes the fake decline.
2. **Difficulty-filter the 38% zero-variance groups**, and drop/repair the known
   scoring-failure dead tasks (banking_4, practice_management_4, tutoring_*).
3. **Target `proper_wrapper` premature-stop / content failures** (decompose
   multi-part tasks, verify each mutation, don't trust empty query results). The
   `direct_mcp_names` slip is now self-recovering and lower priority.

---

## Analysis of `rollouts.jsonl` (2026-06-16)

558 rollouts, 57 scenarios, mean reward 0.411. **Headline: most of the "infra failure" mass is benign — the run-start band is the verifier deliberately being off during warm-up, and the tail band is rollouts force-ended at shutdown. The one real issue is GRPO efficiency: ~28% of full prompt groups are zero-variance and contribute no gradient.** (Revises an earlier draft of this entry that wrongly read the two bands as service outages, and that quoted a `20/50` zero-variance figure that was never computed.)

### Status noise — explained, not problems
- **Run start: front-loaded `judge_error` (129 total, range 0–486 but concentrated at the head) + `no_verifier` (24, range 3–297).** This is the verifier being **intentionally turned off at the start of the run**, not a service that failed to come up. First two tenths score ~0.1 by construction; the band tapers as the verifier comes online. Expected, not a problem.
- **Run tail (lines 516–557): 42 `rollout_error`.** These are active rollout workers **force-ended at run shutdown before they finished** — a teardown artifact, not env failure. Not an issue.
- **`server_error`: 22, scattered across lines 175–515** (not just the tail). Real but minor — likely a database init error. Worth a look but not what's gating the run. *(Note: an earlier draft called this "8"; the actual count is 22, spread through the middle.)*
- **`agent_error`: 12, spread across scenarios (lines 185–513).** Tiny and diffuse — not a systemic model bug this run.

### The real issue — zero-variance groups
Grouping rewards by `(scenario, task_idx)`: groups with identical reward across all generations have zero GRPO advantage → no gradient. This explains a flat/noisy reward curve far better than any per-trajectory pathology.
- **Full groups (size 8 or 16): 12/43 zero-variance (28%)** — the honest measure, excluding the force-ended tail partials.
- **All groups incl. truncated partials: 20/57 (35%)** — the 8 extra are small force-ended groups (size 3–7) that read as zero-variance on too few samples, so they're unreliable; prefer the 28% figure.
- Split (full groups): **8 stuck-hard (all-0.1, never solved)** + **4 trivial (all-1.0, always solved)**. Both ends are wasted compute.
  - Stuck-hard: banking_4#2, billing_and_invoicing_2#7, dating_2#3, form_builder_1#4, messaging_communications_1#9, practice_management_4#3, survey_forms_platform_1#9, tutoring_education_marketplace_1#2.
  - Trivial: clinic_management_2#9, healthcare_telehealth_1#6, identity_and_access_management_1#2, restaurant_operations_1#4.

### Difficulty calibration
Per-group solve-rate (frac of generations scoring 1.0) over the 43 full groups:

| solve-rate | groups | % |
|---|---|---|
| 0% (never) | 10 | 23% |
| (0,25%) | 7 | 16% |
| [25,50%) | 10 | 23% |
| [50,75%) | 4 | 9% |
| [75,100%) | 8 | 19% |
| 100% (always) | 4 | 9% |

- **Mean group solve-rate = 0.40**, close to the GRPO-ideal ~0.5 — the prompt set is *not* badly miscalibrated; the model sits in the productive zone.
- **67% of groups (29/43) have solve/fail spread** → usable advantage signal; only 33% degenerate.
- **Degeneracy is asymmetric: 23% never-solve vs 9% always-solve** — dead-hard tasks drag ~2.5× harder than trivial ones. (Note: "never-solve" is 10 groups but only 8 are strictly zero-variance; the other 2 mix 0.1/0.0, a spurious fail-vs-fail gradient.) So difficulty filtering should weight toward **pruning/replacing the 10 never-solved tasks** over the 4 trivial ones.

### Takeaways
1. **Difficulty-filter the zero-variance groups (28% of full groups)** — drop the always-fail and always-pass prompts so compute lands on groups with reward spread. This is the single highest-value change for GRPO efficiency.
2. The run-start (verifier off) and tail (force-end) bands are **expected/benign** — don't read them as regressions or outages.
3. `server_error` (22, likely DB init) is a real-but-minor follow-up; `agent_error` (12) and the `call_tool` bypass bug are non-issues this run.

---

## 2026-06-17 — rollouts.jsonl (1135 rollouts, async-GRPO run)

### Overview

**1135 rollouts**, mean reward **0.501**. Reward histogram: 510 complete (1.0), 583 incomplete/scoring-failure (0.1), 42 agent_error (0.0).

Statuses: `complete` 510, `incomplete` 467, `agent_error` 42, `judge_error` 59, `server_error` 41, `no_verifier` 16.

Reward by decile (noisy, no clean monotonic trend):
`[0.51, 0.45, 0.51, 0.53, 0.58, 0.61, 0.34, 0.42, 0.45, 0.62, 0.46]`

Per-step reward is highly volatile (e.g. step 0→1 drops 0.10→0.09, step 2→4 rises 0.44→0.89, then step 8 crashes to 0.33). This is expected from the heterogeneous scenario mix — different tasks sampled each step — not a learning pathology.

---

### Difficulty Calibration (Critical)

137 full-size groups (8 or 16 rollouts), 7 truncated partials from run shutdown.

| Bucket | Groups | % |
|---|---|---|
| never (0%) | 47 | 34% |
| (0, 25%) | 10 | 7% |
| [25, 50%) | 17 | 12% |
| [50, 75%) | 14 | 10% |
| [75, 100%) | 16 | 12% |
| always (100%) | 33 | 24% |

**Zero-variance groups: 64/137 (47%)** — these have identical rewards across all rollouts and contribute zero GRPO advantage, producing no gradient. This is the single biggest drag on training efficiency.

- Mean solve-rate: **0.45** (ideal ~0.5 for GRPO)
- Useful (spread) groups: **57/137 (42%)**
- Never-solved: **47 groups**, Always-solved: **33 groups** — ratio 1.42×, meaning dead-hard tasks are the bigger waste relative to always-pass trivials.

**Worst scenarios** (mean reward ≤ 0.1): `accounting_2` (0.062), `b2b_workflow_line_of_business_app_1`, `crm_6`, `iot_smart_infrastructure_management_1`, `membership_management_3`, `job_board_1`, `social_media_5`, `workflow_automation_4` — all 8/8 incomplete.

**Best scenarios** (mean = 1.0, always-pass): `booking_scheduling_8`, `content_bookmark_management_1`, `education_learning_management_2`, `marketplace_9`, and 13 others.

---

### Failure Triage

Of 625 non-complete rollouts:

| Category | Count | % |
|---|---|---|
| proper_wrapper (correct mechanics, wrong answer) | 482 | 77% |
| direct_mcp_names (bypasses call_tool) | 133 | 21% |
| no_tool_calls | 7 | 1% |
| truncated (max_tokens cutoff) | 3 | 0% |

**direct_mcp_names (21%) — driven by think-block truncation**: Model calls env tools by name (e.g. `list_accounts`, `list_zelle_recipients`) instead of through `call_tool`. This rate is elevated vs. prior runs and the root cause is **`max_completion_tokens` being hit mid-think block**: 654/709 assistant messages with text content have an open `<think>` without a closing `</think>` — the model is cut off while reasoning about which tool to call, then force-completion jumps directly to the tool name it had been constructing. 129/144 direct-name calls happen after exactly turn 1 (post-`list_tools`), the exact moment where the model starts a large think block over the tool list. Only 55 rollouts have intact closed think blocks. Fix: **increase `max_completion_tokens`** or **restore `--thinking-token-budget`** to cap individual think block size and prevent runaway thinking from consuming the full token budget before the action can be emitted.

**Scoring failures (10.2% noise)**: 116 rollouts have reward forced to 0.1 — `judge_error` 59, `server_error` 41, `no_verifier` 16. **Almost all are force-terminate artifacts**: the env server is killed mid-rollout (timeout/teardown), causing the verifier or judge to never complete — not actual server instability or judge reliability issues. These are spread throughout the run (not tail-clustered), consistent with per-rollout force-termination rather than end-of-run shutdown. Not a reliability signal to chase.

**Truncation (per prior triage metric) looks low** (3 rollouts flagged by empty-last-message heuristic) but the cut-think-block evidence above shows truncation is the dominant issue — the empty-message heuristic underdetects it when the model is cut mid-`<think>`.

---

### Takeaways

1. **47% zero-variance groups is the top priority.** Nearly half the compute lands on groups with no GRPO gradient. The 47 never-solved groups (34%) are the bigger contributor vs. 33 always-solved (24%). Filtering dead-hard and trivial tasks should be weighted toward the never-solved end.
2. **Increase `max_completion_tokens` or restore `--thinking-token-budget`** — the 21% `direct_mcp_names` rate is elevated because completions are hitting the token ceiling mid-think, causing the model to emit bare tool names. This is an infra/config fix, not a prompt fix.
3. **Scoring failures (10.2%) are force-terminate artifacts** — not judge/server reliability issues. No action needed beyond noting they add noise.
4. Mean solve-rate 0.45 is slightly below ideal 0.5, pulled down by the excess of never-solved tasks. Pruning/replacing those should push the calibration into the ideal band.

---

## 2026-06-23 — rollouts.jsonl + calibration.jsonl (652 rollouts)

**Run scale:** 652 rollouts, 87 groups (72 full at size 8/16, 15 truncated tail
partials from shutdown). Mean reward **0.453**. 12 model versions — a short run
(~12 optimizer steps).

### Reward trend — flat
Reward by tenth: `0.42, 0.52, 0.34, 0.50, 0.56, 0.34, 0.54, 0.47, 0.34, 0.51, 0.50`.
No upward trend. Histogram: `1.0×274 (complete), 0.1×216 (incomplete/scoring-fail),
0.0×162 (agent_error)`.

### Why it's flat: zero-variance + bimodal difficulty
- **Zero-variance: 18/72 full groups (25%)** — identical reward across generations
  → zero GRPO advantage, no gradient. (All-groups count 21/87 is inflated by the
  15 truncated partials; lead with the 25%.) Note this run's 25% is much healthier
  than the 2026-06-17 run's 47%.
- **Difficulty calibration** (full groups): never-solved (0%) **21 (29%)**,
  always-solved (100%) 12 (17%), useful spread 39/72 (54%). Mean solve-rate 0.44
  (ideal ~0.5). **never-solved (21) > always-solved (12)** — dead-hard tasks are
  the bigger drag; filter weighted toward pruning the never-solved end.
- Hard-zero scenarios: `iot_smart_infrastructure_management_1` (0.00),
  `booking_marketplace_5`, `analytics_dashboard_4`, `messaging_communications_1`
  (~0.01–0.03). 12 scenarios sit at exactly 1.000 (mastered).

### calibration.jsonl agrees
72 groups: `learnable 37, uncertain 18, mastered 12, infrastructure_failure 3,
all_failed 1, model_misbehavior 1`. infra failures: `banking_4#2`,
`practice_management_4#3`, `tutoring_education_marketplace_1#2`.

### Failure triage (294 model-fault rollouts: incomplete + agent_error)
| category | count | % |
|---|---|---|
| proper_wrapper (correct mechanics, wrong answer) | 182 | 62% |
| direct_mcp_names (bypasses call_tool) | 106 | 36% |
| no_tool_calls | 6 | 2% |
| truncated (empty last assistant content) | 13 | 4% |

- **direct_mcp_names (36%) mostly self-recovers**: 77/106 retry correctly via
  `call_tool` in the same rollout; only **29 never wrap** and waste the rollout.
  Still **244 wasted direct-name calls** (burned turns even when recovered).
  Unlike the 2026-06-17 run, **truncation is NOT the driver here** — only 13/294
  (4%) end with empty assistant content, so the bare-name calls are not mid-think
  cutoffs. The system prompt already hammers the `call_tool` rule; the prose fix
  has plateaued — needs RL signal or few-shot, not more prose.
- **proper_wrapper (62%)** are genuine task/grounding errors — the real signal.
  E.g. `enterprise_admin_portal_1#4`: cancelled a meal order with
  `validate_cutoff`, got a rejection, accepted it → judged incomplete.

### Scoring failures (infra noise, forced to 0.1)
84/652 (13%): `server_error 45, code_verify_error 17, no_verifier 16,
llm_judge_error 6`. Dilutes variance and inflates the uncertain/zero-variance
counts. `server_error` (45) is the largest non-judged bucket — worth chasing.

### Takeaways
1. Flat curve is **calibration, not per-trajectory pathology**: 25% zero-variance
   + 29% never-solved = large no-gradient fraction. Add difficulty filtering,
   weighted toward dropping never-solved tasks.
2. `direct_mcp_names` self-corrects 73% of the time and is **not** truncation-driven
   this run (contrast 2026-06-17); the prose fix has plateaued.
3. Cut the 13% infra scoring-failure rate (esp. `server_error`) to recover signal.
4. Truncation (4%) and refusals (2%) are not the bottleneck here.

## 2026-06-24 — rollouts.jsonl + calibration.jsonl (repo root)

**Run:** 1024 rollouts, 119 (scenario,task_idx) groups (86 full 8×, 33 shutdown partials).
Companion `calibration.jsonl` = per-task difficulty classification, 109 tasks @ num_rollouts=8.

**Overall reward 0.455.** Histogram: 1.0=432 (complete), 0.1=343 (partial/scoring-fail), 0.0=249 (agent_error). No format violations. Reward-by-tenth peaks mid-run (0.67) then decays to 0.28 at the tail — but the tail is dominated by shutdown partials, so read it as flat-ish, not regression.

**Difficulty calibration (86 full groups):** mean solve-rate **0.47** — well-centered for GRPO. useful/spread = **48/86 (56%)**. Zero-variance = 22/86 (26%). Never-solved 24 (28%) vs always-solved 14 (16%): **dead-hard tasks are the bigger drag** → filtering should weight toward pruning/replacing never-solved over trivial-mastered.

**Calibration file agrees:** learnable 56 (51%), uncertain 26 (24%), mastered 18 (17%), infrastructure_failure 9 (8%). Mean task reward 0.483, mean std 0.224.

**Top actionable: scoring infrastructure, not the policy.** 386/1024 rollouts (38%) scored 0.1 — a chunk is verifier infra, not partial work. Status breakdown: code_verify_error 74, server_error 45, llm_judge_error 10, no_verifier 8. The 9 `infrastructure_failure` tasks (all valid_rollouts=0) are concentrated:
- `code_verify_error` is whole-group on 7 tasks: payments_donations_1#5, social_friends_presence_management_1#0, logistics_management_5#7, iot_smart_home_device_management_2#6, learning_management_system_6#5, project_management_4#5, accounting_finance_1#9, booking_appointments_1#5 — the verifier code itself errors, forcing every rollout to 0.1 (reward_std=0). These contribute zero gradient and silently depress the mean.
- `no_verifier` whole-group on healthcare_patient_portal_5#0.

**Recommendations:**
1. Fix/quarantine the 9 infra tasks (code_verify_error + no_verifier) before next run — they're 8 zero-variance groups masquerading as "hard."
2. Prune/replace the ~24 never-solved groups (asymmetry favors this over cutting trivials).
3. Real policy headroom is the 56% useful split @ mean 0.47 — calibration is healthy once infra noise is removed.

## 2026-06-25 — rollouts.jsonl + calibration.jsonl (repo root, branch exp/awm-dapo)

**Run:** 1037 rollouts, 105 (scenario,task_idx) groups (59 full 8×/16×, 46 shutdown/resample partials). Companion `calibration.jsonl` = 99 tasks @ num_rollouts=8. **This is the DAPO branch** — read the reward distribution accordingly.

**Overall reward 0.269** (down from 0.455 last run, but not comparable — DAPO applies a soft-overlong length penalty, so long trajectories get net-negative reward). Histogram: 1.0=393 (complete), 0.1=249 (partial/scoring-fail), 0.0=150 (agent_error), plus **200 negatives** (74 at exactly −1.0, 126 fractional in the −1<x<0 penalty zone) and a long tail of fractional values. Reward-by-tenth hovers 0.2–0.4 mid-run, craters to −0.37 at the very tail (shutdown partials + accumulated length penalty).

**The negatives are the DAPO length penalty, not format violations or truncation.** Negative-reward rollouts avg **38k completion chars** vs 25k for solved (>0.5); the exact −1.0 bucket is the longest at **43k chars**. Only 5/200 negatives end with an empty assistant message, so this is *not* runaway-`<think>`/`max_tokens` truncation — the model finishes properly but overshoots the soft length budget on long agent_error/incomplete trajectories (neg status: agent_error 108, incomplete 55, code_verify_error 18).

**Difficulty calibration (59 full groups, solve=1.0):** mean solve-rate **0.44** (well-centered for GRPO). useful/spread = **34/59 (58%)**. Zero-variance = 9/59 (15%) full / 12/105 (11%) all. Never-solved **18 (31%)** vs always-solved **7 (12%)**: dead-hard tasks remain the bigger drag → filtering should weight toward pruning/replacing never-solved.

**Calibration file (99 tasks):** learnable 55 (56%), uncertain 22 (22%), mastered 13 (13%), infrastructure_failure 7 (7%), all-failed 1, model_misbehavior 1. Mean task reward 0.334, mean std 0.343.

**Failure triage (609 rollouts ≤0.1):** `proper_wrapper` 384 (63%), **`direct_mcp_names` 224 (37%)**, `no_tool_calls` 1. The wrapper-bypass is still the dominant *mechanical* failure — model calls an MCP tool by name (e.g. `list_robo_profiles`, `create_classified_listing_in_neighborhood`), the env returns a corrective `Unknown tool '...' The only tools you can call directly are ['call_tool','list_tools']`, and the model usually recovers via `call_tool` (e.g. `stock_trading_3#5`) — but it burns a turn each time and pads length (feeding the length penalty above). Mean 4.1 tool-calls/rollout (median 3, max 21).

**Scoring infra noise:** 135/1037 (13%): server_error 56, code_verify_error 39, no_verifier 31, llm_judge_error 9, env_server_error:ConnectionClosedError 1. `server_error` (56) is the largest non-judged bucket and is up vs last run — worth chasing.

**Recommendations:**
1. The mean-reward drop vs 2026-06-24 is mostly the DAPO length penalty biting long trajectories, not a policy regression — judge this run by solve-rate (0.44) and the 58% useful split, both healthy.
2. `direct_mcp_names` (37% of failures) inflates trajectory length → compounds the length penalty. The prose `call_tool` rule has plateaued; needs RL signal/few-shot.
3. **Dynamic sampling (std<1e-8) does NOT catch these.** Of 18 never-solved full groups, only 2 have zero std and get dropped; the other **16 survive the filter** because the soft-overlong length penalty + the 0.0-vs-0.1 partial split give them non-zero reward_std (e.g. practice_management_4#3: [-0.9…0.1], std 0.38). They produce gradient with no actual task signal — pure length-shaping/incomplete-vs-error noise. A *solve-rate*-based difficulty filter would catch them where the *std*-based DAPO filter can't. Prune/replace these 16 (asymmetry still favors this over cutting the 7 trivial-mastered).
4. Cut the 13% scoring-infra rate, esp. `server_error` (56) and `no_verifier` (31).

## 2026-06-27 — rollouts.jsonl (repo root, 114 MB, exp/awm-dapo)

2092 rollouts, **mean reward 0.261**, by-tenth `[0.10, 0.23, 0.25, 0.42, 0.28, 0.18, 0.46, 0.21, 0.21, 0.25, 0.55]` — noisy, no clean trend (last tenth 0.55 is a blip, not a climb). Statuses: complete 825, agent_error 652, incomplete 301, server_error 160, code_verify_error 80, no_verifier 64, llm_judge_error 10.

**Reward histogram is dominated by DAPO overlong shaping.** Beyond the discrete 1.0(742)/0.1(439)/0.0(439) bins there is a large continuous mass of *fractional negative* rewards: 139 at −1.0, 79 at −0.9, plus a long tail across (−0.9, 0). 389 rollouts total are negative. These are the soft-overlong length penalty: negative-reward rollouts average **62.3 KB** completion bytes vs **45.6 KB** for solved (≥0.9) ones — longer ⇒ more penalty, exactly as designed.

**Calibration (79 full groups of 16/32):** mean solve-rate **0.40** (GRPO-ideal ~0.5), **65% useful spread**, buckets never 29% / (0,25%) 20% / [25,50%) 9% / [50,75%) 11% / [75,100%) 25% / always 6%. **never=26 vs always=4** → dead-hard tasks are the dominant drag; a solve-rate difficulty filter should weight toward pruning/replacing the never-solved over the trivial-mastered.

**The length-penalty leak flagged on 2026-06-25 is FIXED (commit `274c87e`, Jun 25, live for this run).** The dynamic-sampling drop now decides on the **base (pre-penalty) verifier reward**, not the length-penalized reward, so the soft-overlong penalty can no longer spread an all-failed group past the std filter. Measured here (base reward reconstructed from `status`): flatness on the *penalized* reward is only **5% (4/79)** — that was my first-pass metric and it understates what the filter drops — but flatness on *base* reward is **18% (14/79)**, and the fixed filter drops **16% (13/79)** (keeping 1 all-mastered group for the "solve and be concise" gradient, by design). So the std-filter is no longer fooled. The residual gap is smaller than the old note implied: of 26 never-solved full groups, **9 are now caught** (base uniformly 0.0); the other **17 survive because base reward genuinely varies (0.0 `agent_error` vs 0.1 `incomplete`/infra-floor)** — that's a real partial-credit split, not a length-penalty artifact. A solve-rate filter would still catch those 17 where the base-std filter can't.

**Two infra problems, both real:**
1. **Tool-backend 500s.** Top tool-result error signatures are all `Error calling create_X. Status code: 500`: create_game 118, create_ticket_type 104, create_course 99, create_product 98, create_membership_plan 70, … These crater specific scenarios — `identity_and_access_management_1` (mean −0.918, 63×500), `workflow_automation_7` (−0.618, 19×500). The model can't complete a task whose `create_*` call 500s.
2. **15% scoring-failure rate** (314/2092: server_error 160, code_verify_error 80, no_verifier 64, llm_judge_error 10). Worse: the overlong length penalty is applied *on top of* the 0.1 scoring-failure floor, pushing **89 of these 314 infra-failures to negative reward** — the policy is being penalized for trajectory length on rollouts the env/verifier failed to score. The length penalty should be masked (or the rollout dropped) when reward_type is a scoring failure.

**Correction to the 2026-06-25 note on `direct_mcp_names`:** in this run, calling tools by their direct name is *not* a pathology. Rollouts with direct-name calls mean reward **0.278** vs wrapper-only **0.251** (neutral-to-better), and the bare-KeyError `'Unknown tool ...'` signature appears only 42×. The generic `call_tool`-wrapper triage from the inspect-rollouts skill does not describe this env — the real wasted turns are the 500s, not wrapper misuse. Truncation (empty last assistant message) is only **3%**, so `max_tokens`/runaway-think is not the problem here.

**Recommendations:**
1. Fix or quarantine the env tool-backend 500s on `create_*` — they alone explain the worst scenarios and a chunk of agent_error.
2. Mask the overlong length penalty on scoring-failure rollouts (89 currently mislabeled negative).
3. Add a **solve-rate** difficulty filter for the residual **17 never-solved groups** the fixed base-std filter still keeps (their base reward varies via the 0.0/0.1 split). The length-penalty leak itself is already fixed (`274c87e`); this is the remaining, smaller gap.
4. Drop the `direct_mcp_names`→needs-signal recommendation for this env; it isn't hurting reward.

## 2026-07-01 — `rollouts.jsonl` (branch `exp/vllm-tp2-pp3`)

1,233 rollouts, 107 (scenario, task_idx) groups. Mean reward **0.403**. This run uses a **graded continuous verifier reward**, not the old discrete 1.0/0.1/0.0 scheme: `complete` spans 0.0–1.0 (mean +0.903) and `incomplete` spans −0.9–0.1 (mean −0.091). Mass spikes at 1.0 (521), 0.1 (435), 0.0 (63), and a **−0.9 floor (84)**. No clear training trend — per-tenth mean bounces 0.33–0.57 with no slope.

**Headline: the −0.9 floor and essentially all negative reward is one failure mode — a runaway, never-closed `<think>` block that gets truncated by `max_tokens`.** 118 rollouts (10%) end on an assistant message with an open `<think>` and no closing `</think>`; they average reward **−0.592** vs **+0.508** for everything else. They explain **79 of 84** floor rollouts and **79 of 145** negatives. Typical trajectory: model calls `list_tools` once, then emits a ~17.7k-char monologue reasoning about which tool to use, runs out of tokens mid-sentence, and never issues a second tool call (floor rollouts average 1.2 tool calls). Note my naive "empty last message" truncation check only caught 24 — these messages are huge, not empty, so length-based detection undercounts; check for unbalanced `<think>` instead.

This is the `--thinking-token-budget` failure from the skill's Part 4b. Fix: cap/budget thinking tokens (or raise `max_tokens` and force a tool call), so the model stops deliberating and acts.

### Difficulty calibration (71 full groups of 8/16)
- Mean solve-rate (reward==1.0) **0.45** — close to the GRPO-ideal 0.5, well calibrated overall.
- **useful (spread) = 37/71 (52%)**; degenerate split is asymmetric: **23 never-solved (32%)** vs **11 always-solved (15%)** → dead-hard tasks are the bigger drag; difficulty filtering should weight toward pruning/replacing the never-solved groups.
- Continuous-reward group std: 17/71 (24%) have ~zero std (no GRPO advantage); mean std 0.275.
- Worst scenarios pulling the mean down: `form_builder_2` (−0.78), `iot_smart_infrastructure_management_1` (−0.70), `job_board_1` (−0.55), `hr_system_1` (−0.47), `booking_platform_6` (−0.42) — these dominate the runaway-think floor hits.

### Other notes
- **`direct_mcp_names` is no longer a content bug:** the env now returns a clear `Unknown tool 'X'. The only tools you can call directly are ['call_tool', 'list_tools']` (the prior KeyError fix landed). Still appears in 239 failures but isn't the driver here.
- Scoring failures ~7%: `no_verifier` (32), `code_verify_error` (42, a new status — graded code verification, mostly −0.9 floor from the same runaway-think dumps), `server_error` (11).
- Truncation aside, mechanics are fine: only 7 `no_tool_calls` refusals among failures.

**Bottom line:** highest-leverage fix is bounding thinking tokens to kill the 10% runaway-`<think>` truncations (−0.592 mean, all the negative reward). Second, prune/replace the 23 never-solved hard groups to recover GRPO signal.

---

## 2026-07-01 — `rollouts.jsonl` + `calibration.jsonl` (repo root, branch `main`, "run #2 upto scenario 200")

558 rollouts over 79 (scenario, task_idx) groups. Mean reward **0.414**. Same graded continuous verifier as the `exp/vllm-tp2-pp3` run: `complete` spans 0.0–1.0 (mean +0.92), `incomplete` spans −0.90–0.10 (mean −0.14). Histogram mass at 1.0 (246), 0.1 (173), 0.0 (36), and a **−0.9 floor (34)** plus a handful at −1.0 (4). Per-tenth reward is noisy with a mild upward tilt (0.55 → dip −0.12 → … → 0.66); no strong slope over this partial run.

**Headline is unchanged: essentially all negative reward is the runaway `<think>` truncation.** 75 rollouts have reward < 0 (mean −0.65); **34 of them end on an assistant message with an open `<think>` and no `</think>`**, and those 34 are exactly the −0.9 floor. Signature trajectory (`clinic_management_2#4`, reward −0.9): one `list_tools` call, then a long `<think>` monologue that runs out of `max_tokens` mid-sentence — no second tool call. Negatives average 3.2 tool calls vs 4.4 for `complete` r=1.0. Fix remains `--thinking-token-budget` (skill Part 4b).

### Difficulty calibration (54 full groups of 8/16)
- Mean solve-rate (reward==1.0) **0.45** — close to GRPO-ideal 0.5, well calibrated overall.
- **useful (spread) = 22/54 (41%)**; degenerate split asymmetric: **20 never-solved (37%)** vs **12 always-solved (22%)** → dead-hard tasks are the bigger drag; weight filtering toward pruning/replacing never-solved groups.
- Zero-variance (strictly identical rewards): **21/54 full groups (39%)** — no GRPO advantage, ~40% of gradient wasted.
- Worst scenarios: `clinic_management_2` (−0.83), `personal_health_records_management_1` (−0.79), `accounting_2` (−0.64), `social_networking_1` (−0.28), `personal_finance_management_1` (−0.26) — these dominate the runaway-think floor.

### calibration.jsonl (running difficulty filter, 58 tasks)
- Precomputed per-task summary built across **model_versions 1–10** (curriculum/difficulty filter running during training), independent of the raw rollouts above (58 tasks vs 79 in rollouts; all 58 present in rollouts).
- Classification mix: **learnable 37, mastered 12, all failed 5, infrastructure_failure 4**. `mastered` (all 8/8 complete) and both failure buckets have zero spread → drop from the train set; the 37 `learnable` carry the signal.
- The 4 `infrastructure_failure` tasks are all `code_verify_error`/`no_verifier` forced to 0.1 (scorer's fault, not the model) — correctly quarantined.

### Other notes
- **`direct_mcp_names` is clean, not a bug:** 39 failing rollouts call an MCP tool by name, but the env returns the explicit `Unknown tool 'X'. The only tools you can call directly are ['call_tool', 'list_tools']` (no more bare KeyError). Wasted turns, not a crash.
- New status **`code_verify_error`** (31): graded code verification; 28 sit at the 0.1 scoring floor, concentrated in 4 tasks (`social_friends_presence_management_1`, `payments_donations_1`, `booking_appointments_1`, `logistics_management_5`) — scorer-side, exclude from training. Plus `no_verifier` (8) and `server_error` (1). Total scoring failures ~7%.
- Mechanics otherwise fine: 0 `no_tool_calls` refusals among failures.

**Bottom line:** same two levers as the prior run — (1) bound thinking tokens to kill the ~6% runaway-`<think>` truncations that account for all negative reward; (2) prune the 20 never-solved + 12 always-solved + 4 infra groups (~40% zero-variance) so GRPO trains on the 22 useful-spread groups.

---

## 2026-07-01 - `rollouts.jsonl` reanalysis with updated learning-signal metrics

Re-ran the updated `inspect-rollouts` view on the repo-root `rollouts.jsonl`: **558 rollouts**, mean reward **0.414**, statuses: complete 287, incomplete 198, code_verify_error 31, agent_error 33, no_verifier 8, server_error 1. Main reward mass: 1.0 x245, 0.1 x173, 0.0 x35, -0.9 x34, -1.0 x4, plus continuous graded values.

### Trajectory trend
- First 100 rollouts: mean **0.225**, complete rate **43%**.
- Last 100 rollouts: mean **0.565**, complete rate **65%**.
- First half: mean **0.365**, complete rate **46.6%**.
- Second half: mean **0.463**, complete rate **56.3%**.

This is a mild upward tilt, but not a clean monotonic learning curve. Per-tenth reward remains task-mix noisy: 0.55, -0.12, 0.41, 0.42, 0.55, 0.17, 0.58, 0.49, 0.39, 0.65, 0.66.

### Learning signal
Full groups only: **54** full `(scenario, task_idx)` groups, **25** partial/non-multiple groups.

- **Strict learnable solve-spread: 22/54 = 40.7%**. These groups contain both solved rollouts (`reward == 1.0`) and not-solved rollouts, so they provide the clearest GRPO contrast.
- **Any reward-spread: 33/54 = 61.1%**. This is looser because it may include partial-credit or infrastructure/noise variation, not necessarily solve/fail contrast.
- **Zero-variance: 21/54 = 38.9%**. These groups provide no GRPO advantage.
- Solve-rate buckets: never solved 20, (0,25%) 5, [25,50%) 4, [50,75%) 2, [75,100%) 11, always solved 12.
- Mean solve-rate: **0.447**, close to the GRPO-ideal 0.5.

`calibration.jsonl` gives a more optimistic curriculum view: **37/58 = 63.8% learnable**, with 12 mastered, 5 all failed, and 4 infrastructure_failure.

### Truncation
Open final `<think>` must be split into harmful vs harmless:

- Open final `<think>` symptom: **49/558 = 8.8%**.
- Harmful open-think with negative reward: **34/558 = 6.1%**.
- Harmless/already-complete open-think: **15/558 = 2.7%**.
- Harmful open-think explains **34/75 = 45.3%** of negative-reward rollouts.

The harmful pattern is: one early tool call, then a very long unclosed `<think>` that hits `max_tokens` before the next action. The harmless pattern is different: the environment state was already correct, so the verifier scored `complete` even though the final reasoning text was unclosed.

### Failure routing
Worst scenarios by mean reward: `clinic_management_2` (-0.826), `personal_health_records_management_1` (-0.786), `accounting_2` (-0.642), `social_networking_1` (-0.275), `personal_finance_management_1` (-0.256). These concentrate the negative/truncation-heavy cases.

Direct MCP-name calls are still common but are not the primary regression signal in this run: 212/558 rollouts have direct-name calls, mean reward **0.445**, complete rate **52.4%**; wrapper-only tool-using rollouts have mean **0.389**, complete rate **50.6%**. Treat direct-name calls as recoverable wasted turns, not the top bug.

Scoring failures remain a filter/masking issue: code_verify_error 31, no_verifier 8, server_error 1. These should not be interpreted as policy failures.

**Bottom line:** the clearer rollout-health numbers are: **40.7% strict learnable full groups**, **38.9% zero-variance full groups**, and **6.1% reward-damaging truncation**. The run has a real but noisy late improvement; the best next levers are still bounding thinking tokens and filtering/replacing never-solved, always-solved, and infrastructure-failure groups.
