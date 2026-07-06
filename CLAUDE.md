# Project Instructions

## Keep Code Simple

- Start with the simplest implementation that works. Do not over-engineer.
- Avoid abstractions, helpers, or utilities unless they are needed by more than one caller.
- Do not add configuration, feature flags, or extensibility hooks speculatively.
- Do not add error handling for cases that cannot happen in practice.
- Three similar lines of code is better than a premature abstraction.
- Add complexity only when the task actually requires it.

## Match the Requested Scope

- Do exactly what was asked — no more. Don't build supporting scaffolding (test
  harnesses, trainers, runners, bootstrapping) the user didn't request. To test a
  function, call the function directly; don't stand up the whole pipeline around it.
- Apply a change at the layer named, not a nearby one. If asked to constrain a
  specific call (e.g. `strategy(board)`, `env.move()`), change that call — not the
  surrounding loop or a different abstraction level.
- Prefer the simplest mechanism that satisfies the request. Don't introduce
  multiprocessing/threading or other machinery unless the user asks for it.
- When the scope is ambiguous, ask before expanding it.

## Package Manager

- Use `uv` for all Python package management (not pip, poetry, or conda).
- Add dependencies with `uv add <package>`.
- Run scripts with `uv run <script>`.

## Available Skills

Skills are defined in `.claude/skills/` and run inline:

- At the start of a task, check whether any project skill matches the request by reading the frontmatter in `.claude/skills/*/SKILL.md`.
- If a skill matches, read its full `SKILL.md` and follow those instructions inline.
- Current skills:
  - `read-arxiv-paper`: Use when asked to read or summarize an arXiv paper from an arXiv URL.
  - `async-grpo-experiment`: Use to auto-run an async-GRPO training run ON the GPU node (two-phase 12→24 steps).
  - `inspect-rollouts`: Use to inspect/summarize/debug a `rollouts.jsonl` trajectory file from an async-GRPO run.
