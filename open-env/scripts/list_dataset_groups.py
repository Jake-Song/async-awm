"""Find the AWM dataset size that covers a target number of unique scenarios.

A "group" is one (scenario, task_idx) prompt — the unit that gets num_generations
rollouts in GRPO. This walks the shuffled dataset in the same order training uses
(build_dataset in openenv_awm_async_grpo.py: flatten one row per task, shuffle
with seed 42) starting at --start, counting unique scenarios as it goes, and
stops once that count reaches --num-scenarios. It returns the dataset size: the
number of groups scanned to reach that many unique scenarios.

Requires the AWM env server running (environment.url in config.yaml). The training
module isn't imported here to avoid pulling in trl/torch just to list rows.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from datasets import Dataset

from openenv.core.env_server.mcp_types import CallToolAction
from agent_world_model_env import AWMEnv

CONFIG_PATH = Path(__file__).resolve().parent.parent / "open-env" / "config.yaml"


def dataset_size(env_url: str, num_scenarios: int, start: int) -> tuple[int, int]:
    """Scan the shuffled dataset from `start` until `num_scenarios` unique
    scenarios are seen. Return (unique_reached, groups_scanned)."""
    env = AWMEnv(base_url=env_url).sync()
    with env:
        result = env.step(CallToolAction(tool_name="__list_scenarios__", arguments={}))
        scenarios = result.observation.scenarios

    scenario_names, task_indices = [], []
    for scenario in scenarios:
        for task_idx, _ in enumerate(scenario["tasks"]):
            scenario_names.append(scenario["name"])
            task_indices.append(task_idx)

    # Same shuffle as build_dataset (seed=42) so the order matches training.
    shuffled = Dataset.from_dict(
        {"scenario": scenario_names, "task_idx": task_indices}
    ).shuffle(seed=42)["scenario"]

    seen: set[str] = set()
    groups = 0
    for name in shuffled[start:]:
        seen.add(name)
        groups += 1
        if len(seen) >= num_scenarios:
            break
    return len(seen), groups


def parse_args() -> argparse.Namespace:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-url", default=config["environment"]["url"])
    parser.add_argument("--num-scenarios", type=int, default=250,
                        help="Target number of unique scenarios to cover.")
    parser.add_argument("--start", type=int, default=0,
                        help="Offset into the shuffled dataset.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_scenarios < 1 or args.start < 0:
        raise SystemExit("--num-scenarios must be >= 1 and --start >= 0")

    reached, size = dataset_size(args.env_url, args.num_scenarios, args.start)
    print(f"unique scenarios reached: {reached} (target={args.num_scenarios})")
    print(f"dataset size:             {size} (groups from start={args.start})")


if __name__ == "__main__":
    main()
