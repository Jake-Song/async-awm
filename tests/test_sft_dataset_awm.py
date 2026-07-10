from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from awm.sft_dataset_awm import (
    WRAPPER_TOOLS,
    collect_task_records,
    derive_sft_rows,
    normalize_assistant_message,
    rebuild_sft_dataset,
    to_openai_messages,
    validate_dataset,
    validate_sft_row,
)
from awm.trajectory_utils import windowed_messages


def assistant_call(call_id: str, name: str, arguments: dict) -> dict:
    return {
        "role": "assistant",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        ],
    }


def tool_result(call_id: str, name: str, content: str = "ok") -> dict:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": name,
        "content": content,
    }


def successful_episode() -> dict:
    prompt = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "do the task"},
    ]
    completion = [
        assistant_call("list-1", "list_tools", {}),
        tool_result("list-1", "list_tools", "catalog"),
        assistant_call(
            "bad-1",
            "call_tool",
            {"tool_name": "update", "arguments": {"bad": True}},
        ),
        tool_result("bad-1", "call_tool", "Error: invalid arguments"),
        assistant_call(
            "good-1",
            "call_tool",
            {"tool_name": "update", "arguments": {"id": 1}},
        ),
        tool_result("good-1", "call_tool", "updated"),
        {"role": "assistant", "content": "Done."},
    ]
    return {
        "schema_version": 1,
        "episode_id": "scenario#0@1",
        "scenario": "scenario",
        "task_idx": 0,
        "task": "do the task",
        "attempt": 1,
        "reward": 1.0,
        "status": "complete",
        "eligible": True,
        "eligibility_reason": "verified_complete",
        "structural_errors": [],
        "messages": prompt + completion,
        "tools": WRAPPER_TOOLS,
        "turns": [
            {"turn_index": 0, "completion_index": 0, "trainable": True},
            {"turn_index": 1, "completion_index": 2, "trainable": False},
            {"turn_index": 2, "completion_index": 4, "trainable": True},
            {"turn_index": 3, "completion_index": 6, "trainable": True},
        ],
        "metrics": {"assistant_turns": 4},
    }


class MessageTests(unittest.TestCase):
    def test_window_pins_list_tools_and_keeps_recent_turn(self):
        prompt = [{"role": "system"}, {"role": "user"}]
        completion = []
        for index in range(4):
            name = "list_tools" if index == 0 else "call_tool"
            completion.extend(
                [assistant_call(str(index), name, {}), tool_result(str(index), name)]
            )

        result = windowed_messages(prompt, completion, context_window_turns=1)

        self.assertEqual(result, prompt + completion[:2] + completion[-2:])

    def test_openai_arguments_are_normalized_to_dict_and_back_to_json(self):
        assistant, errors = normalize_assistant_message(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "call_tool",
                            "arguments": '{"tool_name":"x","arguments":{"id":1}}',
                        },
                    }
                ],
            },
            fallback_call_id_prefix="fallback",
        )

        self.assertEqual(errors, [])
        arguments = assistant["tool_calls"][0]["function"]["arguments"]
        self.assertEqual(arguments["arguments"], {"id": 1})
        api_message = to_openai_messages([assistant])[0]
        self.assertIsInstance(
            api_message["tool_calls"][0]["function"]["arguments"], str
        )

    def test_success_is_split_into_trainable_turns_and_bad_target_is_skipped(self):
        rows = derive_sft_rows(successful_episode(), context_window_turns=1)

        self.assertEqual([row["turn_index"] for row in rows], [0, 2, 3])
        recovery_prompt = rows[1]["prompt"]
        self.assertIn(
            "Error: invalid arguments", [m.get("content") for m in recovery_prompt]
        )
        final_prompt = rows[2]["prompt"]
        self.assertNotIn(
            "Error: invalid arguments", [m.get("content") for m in final_prompt]
        )

    def test_sft_validator_rejects_non_object_nested_arguments(self):
        row = derive_sft_rows(successful_episode(), context_window_turns=3)[0]
        row["completion"][0] = assistant_call(
            "bad", "call_tool", {"tool_name": "x", "arguments": "not-an-object"}
        )
        self.assertIn("call_tool.arguments must be an object", validate_sft_row(row))


class CollectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_retry_stops_on_success_and_skips_existing_success(self):
        tasks = [
            {"scenario": "a", "task_idx": 0},
            {"scenario": "b", "task_idx": 0},
        ]
        records = []

        async def run_attempt(task, attempt):
            return {
                "scenario": task["scenario"],
                "task_idx": task["task_idx"],
                "attempt": attempt,
                "eligible": attempt == 2,
            }

        async def on_record(record):
            records.append(record)

        progress = await collect_task_records(
            tasks,
            previous_attempts={},
            successful_tasks={("b", 0)},
            max_attempts=3,
            concurrency=2,
            run_attempt=run_attempt,
            on_record=on_record,
        )

        self.assertEqual(
            [(r["scenario"], r["attempt"]) for r in records], [("a", 1), ("a", 2)]
        )
        self.assertEqual(progress.attempted, 2)
        self.assertEqual(progress.succeeded, 1)
        self.assertEqual(progress.exhausted, 0)


class DatasetValidationTests(unittest.TestCase):
    def test_rebuild_and_validate_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            episode = successful_episode()
            (output_dir / "episodes.jsonl").write_text(
                json.dumps(episode) + "\n", encoding="utf-8"
            )
            (output_dir / "metadata.json").write_text(
                json.dumps({"config": {"context_window_turns": 1}}),
                encoding="utf-8",
            )

            tasks, rows = rebuild_sft_dataset(
                output_dir / "episodes.jsonl",
                output_dir / "sft.jsonl",
                context_window_turns=1,
            )
            report = validate_dataset(output_dir)

            self.assertEqual((tasks, rows), (1, 3))
            self.assertTrue(report["valid"], report["errors"])
            self.assertEqual(report["sft_rows"], 3)


if __name__ == "__main__":
    unittest.main()
