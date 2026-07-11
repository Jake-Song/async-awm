import argparse
import json


def readable(value, indent=0):
    """Render JSON-like output without escaping multiline text fields."""
    padding = " " * indent

    if isinstance(value, dict):
        if not value:
            return "{}"
        items = []
        for key, item in value.items():
            rendered = readable(item, indent + 2)
            items.append(f'{" " * (indent + 2)}{json.dumps(str(key))}: {rendered}')
        return "{\n" + ",\n".join(items) + f"\n{padding}}}"

    if isinstance(value, list):
        if not value:
            return "[]"
        items = [f'{" " * (indent + 2)}{readable(item, indent + 2)}' for item in value]
        return "[\n" + ",\n".join(items) + f"\n{padding}]"

    if isinstance(value, str) and "\n" in value:
        block_indent = " " * (indent + 2)
        body = "\n".join(f"{block_indent}{line}" for line in value.splitlines())
        return f'"""\n{body}\n{padding}"""'

    return json.dumps(value, ensure_ascii=False)


def judge_result(record):
    """Return the judge result from either a payload or repeated-vote record."""
    sql_verifier = record.get("sql_verifier") or {}
    return sql_verifier.get("original_judge") or record.get("judge_result") or {}


def judge_view(record, line_number):
    result = judge_result(record)
    classification = result.get("classification") or record.get("classification") or "unknown"
    source_status = record.get("source_status")
    replay_status = (record.get("sql_verifier") or {}).get("reward_type")

    heading = f"[{line_number}] {record.get('scenario', '?')}#{record.get('task_idx', '?')}"
    lines = [heading]
    if record.get("source_rollout_line") is not None:
        lines.append(f"Source rollout line: {record['source_rollout_line']}")
    if source_status is not None:
        lines.append(f"Source result: {source_status} ({record.get('source_reward', 'n/a')})")
    if replay_status is not None:
        lines.append(f"Replay result: {replay_status} ({(record.get('sql_verifier') or {}).get('reward', 'n/a')})")
    if record.get("model") is not None:
        lines.append(f"Judge: {record['model']} (vote {record.get('vote_idx', '?')})")
    lines.append(f"Classification: {classification}")

    confidence = result.get("confidence_score")
    if confidence is not None:
        labels = ("complete", "incomplete", "server_error", "agent_error")
        if isinstance(confidence, list) and len(confidence) == len(labels):
            confidence = ", ".join(
                f"{label}={score}" for label, score in zip(labels, confidence, strict=True)
            )
        lines.append(f"Confidence: {confidence}")

    if record.get("collection_error"):
        lines.extend(("", "Collection error:", str(record["collection_error"])))
        return "\n".join(lines)

    if record.get("task"):
        lines.extend(("", "Task:", str(record["task"])))
    lines.extend(("", "Judge reasoning:", str(result.get("reasoning") or "No reasoning recorded.")))
    if result.get("evidence"):
        lines.extend(("", "Evidence:", readable(result["evidence"])))
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description="Pretty-print records from a JSONL file")
    parser.add_argument("path")
    parser.add_argument("line", nargs="?", type=int, help="physical line number to print")
    output = parser.add_mutually_exclusive_group()
    output.add_argument("-b", "--brief", action="store_true")
    output.add_argument(
        "--judge",
        action="store_true",
        help="show a compact LLM-judge view for payload or vote records",
    )
    output.add_argument(
        "--json",
        action="store_true",
        help="emit strict JSON (multiline strings will contain escaped newlines)",
    )
    parser.add_argument("--scenario", help="only show records from this scenario")
    parser.add_argument("--task-idx", type=int, help="only show this task index")
    parser.add_argument(
        "--classification", help="only show this judge classification (with --judge)"
    )
    parser.add_argument(
        "--disagreements",
        action="store_true",
        help="only show source/judge classification disagreements (with --judge)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.path) as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line or (args.line is not None and line_number != args.line):
                continue

            record = json.loads(line)
            if args.scenario is not None and record.get("scenario") != args.scenario:
                continue
            if args.task_idx is not None and record.get("task_idx") != args.task_idx:
                continue

            result = judge_result(record) if args.judge else {}
            classification = result.get("classification") or record.get("classification")
            if args.classification is not None and classification != args.classification:
                continue
            if args.disagreements:
                source_status = record.get("source_status")
                if source_status is None or source_status == classification:
                    continue

            if args.brief:
                record = {
                    key: record[key]
                    for key in ("scenario", "task_idx", "reward", "status")
                }

            if args.judge:
                print(judge_view(record, line_number))
            elif args.json:
                print(json.dumps(record, indent=2, ensure_ascii=False))
            else:
                print(readable(record))
            print("-" * 80)


if __name__ == "__main__":
    main()
