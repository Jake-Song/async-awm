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


def parse_args():
    parser = argparse.ArgumentParser(description="Pretty-print records from a JSONL file")
    parser.add_argument("path")
    parser.add_argument("line", nargs="?", type=int, help="physical line number to print")
    parser.add_argument("-b", "--brief", action="store_true")
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit strict JSON (multiline strings will contain escaped newlines)",
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
            if args.brief:
                record = {
                    key: record[key]
                    for key in ("scenario", "task_idx", "reward", "status")
                }

            if args.json:
                print(json.dumps(record, indent=2, ensure_ascii=False))
            else:
                print(readable(record))
            print("-" * 80)


if __name__ == "__main__":
    main()
