"""Convert BFCL_v4_multi_turn_base data into the AWM prompt/response format.

The AWM rollout format (see `rollouts.jsonl`) is one record per
`(scenario, task_idx)` whose `prompt` is a chat message list beginning with the
AWM MCP system prompt. A BFCL `multi_turn_base` entry packs several *sequential*
user turns into one episode. We collapse the whole episode into a single turn:
all user turns are concatenated into one user message, emitting one AWM record
per BFCL entry (task_idx 0) plus the shared episode config so the env state is
reconstructable.

Usage:
    uv run experiment/bfcl_to_awm.py \
        --in experiment/bfcl_data_examples.jsonl \
        --out experiment/bfcl_awm.jsonl
"""

import argparse
import json

# Canonical AWM system prompt, copied verbatim from
# OpenEnv/envs/agent_world_model_env/server/prompts.py (DEFAULT_SYSTEM_PROMPT).
SYSTEM_PROMPT = """\
You are at a MCP environment. You need to call MCP tools to assist with the user query. \
At each step, you can only call one function. You have already logged in, and your user id is 1 if required.

You are provided with TWO functions:

1. list_tools
   - Description: List all available MCP tools for the current environment.
   - Arguments: None

2. call_tool
   - Description: Call a MCP environment-specific tool
   - Arguments:
       - tool_name: str, required
       - arguments: str, required, valid JSON string

For each function call, return a json object within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

Example:
<tool_call>
{"name": "call_tool", "arguments": {"tool_name": "get_weather", "arguments": "{\\"city\\": \\"Beijing\\"}"}}
</tool_call>

You should call list_tools first to discover available tools, then use call_tool to interact. \
When you have enough information to answer, output the answer directly without any tool_call tags."""


def convert_entry(entry: dict) -> dict:
    """Turn one BFCL multi_turn_base entry into a single-turn AWM record."""
    scenario = entry["id"]
    turns = entry["question"]

    # Sum the episode into one turn: concatenate every user message across all
    # turns into a single user prompt.
    contents = [msg["content"] for turn in turns for msg in turn]
    combined = "\n".join(contents)
    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": combined},
    ]

    return {
        "scenario": scenario,
        "task_idx": 0,
        "prompt": prompt,
        "completion": [],
        "involved_classes": entry.get("involved_classes", []),
        "ground_truth": entry.get("path", []),
        "initial_config": entry.get("initial_config", {}),
        "num_turns": len(turns),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="inp", default="experiment/bfcl_data_examples.jsonl")
    parser.add_argument("--out", dest="out", default="experiment/bfcl_awm.jsonl")
    args = parser.parse_args()

    n_entries = 0
    n_records = 0
    with open(args.inp) as fin, open(args.out, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_entries += 1
            record = convert_entry(json.loads(line))
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_records += 1

    print(f"Converted {n_entries} BFCL entries -> {n_records} AWM records: {args.out}")


if __name__ == "__main__":
    main()
