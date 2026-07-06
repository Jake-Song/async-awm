import json
import sys

args = sys.argv[1:]
brief = "-b" in args
args = [a for a in args if a != "-b"]
path = args[0]
target = int(args[1]) if len(args) > 1 else None
with open(path) as f:
    for i, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        if target is not None and i != target:
            continue
        r = json.loads(line)
        if brief:
            r = {k: r[k] for k in ("scenario", "task_idx", "reward", "status")}
        print(json.dumps(r, indent=2, ensure_ascii=False))
        print("-" * 80)
