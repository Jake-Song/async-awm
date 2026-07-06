"""Simulate sampler/trainer allocation on a fixed GPU budget.

The model assumes each sampling server and each training process uses one GPU.
Sampling completions are staggered across a shared pool. The distributed trainer
starts an optimization step as soon as its global batch is available.
"""

from __future__ import annotations

import argparse
from collections import deque


def simulate(
    sampling_servers: int,
    training_servers: int,
    dataset_groups: int,
    num_generations: int,
    target_training_steps: int,
    samples_per_sampling_batch: int,
    sampling_batch_seconds: int,
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
    training_step_seconds: int,
    max_inflight_tasks: int,
    max_staleness: int,
) -> dict[str, int]:
    queued_samples = 0
    queued_batches: deque[list[int]] = deque()
    produced_samples = 0
    produced_groups = 0
    partial_group_samples = 0
    dropped_samples = 0
    completed_steps = 0
    trainer_idle_seconds = 0
    training_time_left = 0
    samples_per_step = (
        per_device_batch_size
        * gradient_accumulation_steps
        * training_servers
    )
    sampling_slots = min(
        sampling_servers * samples_per_sampling_batch,
        max_inflight_tasks,
    )
    sampling_time_left = [
        1 + slot * sampling_batch_seconds // sampling_slots
        for slot in range(sampling_slots)
    ]
    elapsed_seconds = 0

    while completed_steps < target_training_steps:
        elapsed_seconds += 1
        if training_time_left > 0:
            training_time_left -= 1
            if training_time_left == 0:
                completed_steps += 1
                if completed_steps >= target_training_steps:
                    break

        for slot in range(sampling_slots):
            sampling_time_left[slot] -= 1
            if sampling_time_left[slot] == 0 and produced_groups < dataset_groups:
                partial_group_samples += 1
                if partial_group_samples == num_generations:
                    produced_groups += 1
                    produced_samples += num_generations
                    queued_samples += num_generations
                    if queued_batches and queued_batches[-1][0] == completed_steps:
                        queued_batches[-1][1] += num_generations
                    else:
                        queued_batches.append([completed_steps, num_generations])
                    partial_group_samples = 0
                sampling_time_left[slot] = sampling_batch_seconds

        while (
            queued_batches
            and completed_steps - queued_batches[0][0] > max_staleness
        ):
            dropped_samples += queued_batches[0][1]
            queued_samples -= queued_batches[0][1]
            queued_batches.popleft()

        if (
            completed_steps < target_training_steps
            and training_time_left == 0
            and queued_samples >= samples_per_step
        ):
            queued_samples -= samples_per_step
            samples_needed = samples_per_step
            while samples_needed > 0:
                samples_used = min(samples_needed, queued_batches[0][1])
                queued_batches[0][1] -= samples_used
                samples_needed -= samples_used
                if queued_batches[0][1] == 0:
                    queued_batches.popleft()
            training_time_left = training_step_seconds

        if training_time_left == 0:
            trainer_idle_seconds += training_servers

        if (
            training_time_left == 0
            and queued_samples < samples_per_step
            and (produced_groups >= dataset_groups or sampling_slots == 0)
        ):
            break

    return {
        "training_steps": completed_steps,
        "elapsed_seconds": elapsed_seconds,
        "trainer_idle_seconds": trainer_idle_seconds,
        "produced_groups": produced_groups,
        "produced_samples": produced_samples,
        "dropped_samples": dropped_samples,
        "queued_samples": queued_samples,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find the best sampler/trainer split for a fixed GPU budget."
    )
    parser.add_argument("--total-gpus", type=int, default=8)
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=107,
        help="Number of prompt groups in the dataset.",
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=8,
        help="Individual rollout samples generated for each prompt group.",
    )
    parser.add_argument("--samples-per-sampling-batch", type=int, default=60)
    parser.add_argument("--sampling-batch-seconds", type=int, default=370)
    parser.add_argument("--per-device-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--training-step-seconds", type=int, default=25)
    parser.add_argument(
        "--max-inflight-tasks",
        type=int,
        default=-1,
        help="Maximum concurrent sampling tasks; -1 uses the TRL calculation.",
    )
    parser.add_argument("--max-staleness", type=int, default=6)
    parser.add_argument(
        "--gpu-hour-price",
        type=float,
        default=3.29,
        help="USD per GPU-hour; cost bills all --total-gpus for the wall-clock run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.total_gpus < 2:
        raise SystemExit("--total-gpus must be at least 2")

    allocations = [
        (sampling, args.total_gpus - sampling)
        for sampling in range(1, args.total_gpus)
    ]

    positive_values = (
        args.dataset_size,
        args.num_generations,
        args.samples_per_sampling_batch,
        args.sampling_batch_seconds,
        args.per_device_batch_size,
        args.gradient_accumulation_steps,
        args.training_step_seconds,
    )
    if min(positive_values) < 1:
        raise SystemExit("dataset size, sample counts, and timings must be positive")
    if args.max_staleness < 0:
        raise SystemExit("--max-staleness must be non-negative")
    if args.max_inflight_tasks == 0 or args.max_inflight_tasks < -1:
        raise SystemExit("--max-inflight-tasks must be -1 or positive")

    results = []
    for sampling_servers, training_servers in allocations:
        samples_per_step = (
            args.per_device_batch_size
            * args.gradient_accumulation_steps
            * training_servers
        )
        dataset_steps = (
            args.dataset_size * args.num_generations // samples_per_step
        )
        max_inflight_tasks = args.max_inflight_tasks
        if max_inflight_tasks == -1:
            max_inflight_tasks = (
                args.max_staleness
                * args.per_device_batch_size
                * args.gradient_accumulation_steps
                * training_servers
            )
        result = simulate(
            sampling_servers=sampling_servers,
            training_servers=training_servers,
            dataset_groups=args.dataset_size,
            num_generations=args.num_generations,
            target_training_steps=dataset_steps,
            samples_per_sampling_batch=args.samples_per_sampling_batch,
            sampling_batch_seconds=args.sampling_batch_seconds,
            per_device_batch_size=args.per_device_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            training_step_seconds=args.training_step_seconds,
            max_inflight_tasks=max_inflight_tasks,
            max_staleness=args.max_staleness,
        )
        result["sampling_servers"] = sampling_servers
        result["training_servers"] = training_servers
        result["max_inflight_tasks"] = max_inflight_tasks
        result["dataset_steps"] = dataset_steps
        results.append(result)

    print(
        "sampling  training  max_inflight  target_steps  training_steps  "
        "elapsed_hours  idle_hours_per_gpu  idle_fraction  groups_made  "
        "samples_made  dropped_samples  samples_queued  cost_usd"
    )
    for result in results:
        idle_hours_per_gpu = (
            result["trainer_idle_seconds"] / result["training_servers"] / 3600
        )
        idle_fraction = 0.0
        if result["elapsed_seconds"] > 0:
            idle_fraction = (
                result["trainer_idle_seconds"]
                / result["training_servers"]
                / result["elapsed_seconds"]
            )
        cost_usd = (
            result["elapsed_seconds"] / 3600
            * args.total_gpus
            * args.gpu_hour_price
        )
        print(
            f"{result['sampling_servers']:>8}  "
            f"{result['training_servers']:>8}  "
            f"{result['max_inflight_tasks']:>12}  "
            f"{result['dataset_steps']:>13}  "
            f"{result['training_steps']:>14}  "
            f"{result['elapsed_seconds'] / 3600:>13.2f}  "
            f"{idle_hours_per_gpu:>18.2f}  "
            f"{idle_fraction:>13.2f}  "
            f"{result['produced_groups']:>11}  "
            f"{result['produced_samples']:>12}  "
            f"{result['dropped_samples']:>15}  "
            f"{result['queued_samples']:>14}  "
            f"{cost_usd:>8.2f}"
        )

    finished = [
        result
        for result in results
        if result["training_steps"] >= result["dataset_steps"]
    ]
    if finished:
        best = min(finished, key=lambda result: result["elapsed_seconds"])
        best_cost = (
            best["elapsed_seconds"] / 3600 * args.total_gpus * args.gpu_hour_price
        )
        print(
            "\nBest allocation: "
            f"{best['sampling_servers']} sampling servers + "
            f"{best['training_servers']} training servers "
            f"({best['training_steps']} training steps in "
            f"{best['elapsed_seconds'] / 3600:.2f} hours, ${best_cost:,.2f})"
        )
    else:
        print("\nNo allocation could complete the dataset.")


if __name__ == "__main__":
    main()
