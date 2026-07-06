"""Simulate the LLM-judge cost of an AWM async-GRPO run.

The AWM environment fires exactly one LLM-judge call per `verify` step in SQL
mode (agent_world_model_env/server/verifier.py: run_llm_judge); the code
verifier is deterministic and free. With verifier_mode=sql, every rollout that
gets verified costs one judge call, so the run total is

    dataset_size * num_generations * num_epochs * sql_fraction

calls (defaults mirror open-env/config.yaml: 1000 groups, 8 generations,
1 epoch, sql verifier). Each call is priced by sampling its input tokens (fixed
judge system prompt + verification_json + the agent trajectory, which dominates
and scales with rollout turns) and its output tokens (the JSON verdict, capped
at the verifier's max_completion_tokens=4096).

Prices default to placeholders — set --price-in / --price-out to the real
$/1M-token rates for the model behind verifier.llm_model / OPENENV_AWM_LLM_MODEL.
"""

from __future__ import annotations

import argparse

import numpy as np

# verifier.py hard-caps the judge response at this many tokens.
JUDGE_MAX_TOKENS = 4096


def _lognormal(rng: np.random.Generator, median: float, sigma: float, size: int):
    """Positive, right-skewed samples with the given median (sigma in log-space)."""
    return rng.lognormal(mean=np.log(median), sigma=sigma, size=size)


def simulate(
    judge_calls: int,
    system_tok: int,
    median_turns: float,
    turns_sigma: float,
    max_turns: int,
    tokens_per_turn: float,
    median_verification_tok: float,
    median_output_tok: float,
    price_in: float,
    price_out: float,
    trials: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    price_in_tok = price_in / 1e6
    price_out_tok = price_out / 1e6

    costs = np.empty(trials)
    in_means = np.empty(trials)
    out_means = np.empty(trials)

    for t in range(trials):
        n = judge_calls

        # Input = judge system prompt (fixed) + verification_json + trajectory.
        # Trajectory tokens scale with agent turns, capped at rollout.max_turns.
        turns = np.minimum(_lognormal(rng, median_turns, turns_sigma, n), max_turns)
        trajectory = turns * tokens_per_turn
        verification = _lognormal(rng, median_verification_tok, 0.5, n)
        input_tok = system_tok + verification + trajectory

        # Output verdict, capped at the verifier's max_completion_tokens.
        output_tok = np.minimum(
            _lognormal(rng, median_output_tok, 0.5, n), JUDGE_MAX_TOKENS
        )

        costs[t] = float(np.sum(input_tok * price_in_tok + output_tok * price_out_tok))
        in_means[t] = input_tok.mean()
        out_means[t] = output_tok.mean()

    p10, p50, p90 = np.percentile(costs, [10, 50, 90])
    return {
        "mean_input_tok": float(in_means.mean()),
        "mean_output_tok": float(out_means.mean()),
        "cost_mean": float(costs.mean()),
        "cost_p10": float(p10),
        "cost_p50": float(p50),
        "cost_p90": float(p90),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate the SQL-verifier LLM-judge cost of an AWM run."
    )
    # Run shape (defaults from open-env/config.yaml).
    parser.add_argument("--dataset-size", type=int, default=300,
                        help="Number of prompt groups in the dataset.")
    parser.add_argument("--num-generations", type=int, default=8,
                        help="Rollout samples generated per prompt group.")
    parser.add_argument("--num-epochs", type=int, default=1,
                        help="Passes over the dataset (training.num_epochs).")
    parser.add_argument("--sql-fraction", type=float, default=1.0,
                        help="Fraction of rollouts verified in SQL mode (code mode is free).")
    # Token model (trajectory knobs mirror rollout.* in config.yaml).
    parser.add_argument("--system-tok", type=int, default=800,
                        help="Fixed judge system-prompt tokens.")
    parser.add_argument("--max-turns", type=int, default=20,
                        help="Cap on agent turns per rollout (rollout.max_turns).")
    parser.add_argument("--median-turns", type=float, default=4.0,
                        help="Median agent turns per rollout.")
    parser.add_argument("--turns-sigma", type=float, default=0.5,
                        help="Log-space spread of rollout length.")
    parser.add_argument("--tokens-per-turn", type=float, default=4096.0,
                        help="Trajectory tokens contributed per agent turn "
                             "(rollout.max_completion_length).")
    parser.add_argument("--median-verification-tok", type=float, default=600.0,
                        help="Median verification_json tokens.")
    parser.add_argument("--median-output-tok", type=float, default=500.0,
                        help="Median verdict tokens (capped at 4096).")
    # Pricing ($ per 1M tokens) — set to the judge model's real rates.
    parser.add_argument("--price-in", type=float, default=0.09,
                        help="$ per 1M input tokens (PLACEHOLDER).")
    parser.add_argument("--price-out", type=float, default=0.18,
                        help="$ per 1M output tokens (PLACEHOLDER).")
    # Monte Carlo.
    parser.add_argument("--trials", type=int, default=2000,
                        help="Monte Carlo replicates of the whole run.")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    positive = (
        args.dataset_size, args.num_generations, args.num_epochs,
        args.max_turns, args.median_turns, args.tokens_per_turn,
    )
    if min(positive) < 1:
        raise SystemExit("dataset/generation/epoch counts and token rates must be positive")
    if not 0.0 <= args.sql_fraction <= 1.0:
        raise SystemExit("--sql-fraction must be in [0, 1]")

    judge_calls = round(
        args.dataset_size * args.num_generations * args.num_epochs * args.sql_fraction
    )

    result = simulate(
        judge_calls=judge_calls,
        system_tok=args.system_tok,
        median_turns=args.median_turns,
        turns_sigma=args.turns_sigma,
        max_turns=args.max_turns,
        tokens_per_turn=args.tokens_per_turn,
        median_verification_tok=args.median_verification_tok,
        median_output_tok=args.median_output_tok,
        price_in=args.price_in,
        price_out=args.price_out,
        trials=args.trials,
        seed=args.seed,
    )

    print(f"Judge calls per run:     {judge_calls:,}")
    print(f"  = dataset {args.dataset_size} x gen {args.num_generations} "
          f"x epochs {args.num_epochs} x sql {args.sql_fraction}")
    print(f"Mean input tokens/call:  {result['mean_input_tok']:,.0f}")
    print(f"Mean output tokens/call: {result['mean_output_tok']:,.0f}")
    print(f"Prices: ${args.price_in}/1M in, ${args.price_out}/1M out")
    print("-" * 40)
    print(f"Cost per run  mean: ${result['cost_mean']:,.2f}")
    print(f"              p10:  ${result['cost_p10']:,.2f}")
    print(f"              p50:  ${result['cost_p50']:,.2f}")
    print(f"              p90:  ${result['cost_p90']:,.2f}")
    print(f"Cost per judge call (mean): ${result['cost_mean'] / judge_calls:.4f}")


if __name__ == "__main__":
    main()
