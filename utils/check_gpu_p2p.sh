#!/usr/bin/env bash
set -uo pipefail

# Check whether GPU peer-to-peer (P2P) access is working between NVIDIA GPUs.
# P2P lets one GPU read/write another's memory directly over PCIe/NVLink and is
# what NCCL uses to transfer trainer weights into the vLLM server in the 2-GPU
# GRPO setup. Reports both the driver's advertised P2P capability and a real
# functional test (an actual cross-GPU tensor copy via PyTorch).

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found - no NVIDIA driver/GPUs available." >&2
    exit 1
fi

gpu_count="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
echo "Detected $gpu_count GPU(s):"
nvidia-smi --query-gpu=index,name --format=csv,noheader | sed 's/^/  GPU /'
echo

if [ "$gpu_count" -lt 2 ]; then
    echo "Only $gpu_count GPU detected - P2P needs at least 2 GPUs. Nothing to test."
    exit 0
fi

echo "=== Topology matrix (nvidia-smi topo -m) ==="
nvidia-smi topo -m
echo

# P2P capability matrix. Available on newer drivers; "OK" = supported,
# "CNS" = chipset not supported, "NS" = not supported.
echo "=== P2P read capability (nvidia-smi topo -p2p r) ==="
if ! nvidia-smi topo -p2p r 2>/dev/null; then
    echo "(driver does not support 'topo -p2p' query; relying on functional test below)"
fi
echo
echo "=== P2P write capability (nvidia-smi topo -p2p w) ==="
nvidia-smi topo -p2p w 2>/dev/null || true
echo

# Functional test: actually copy a tensor between every ordered GPU pair and
# verify the data survives the trip. can_device_access_peer reports the driver's
# view; the copy proves the path really works.
echo "=== Functional P2P test (PyTorch cross-GPU copy) ==="
if ! command -v uv >/dev/null 2>&1; then
    echo "(uv not found; skipping functional PyTorch test)"
    exit 0
fi

python - <<'PY'
import sys

try:
    import torch
except ModuleNotFoundError:
    print("(torch not installed; skipping functional test)")
    sys.exit(0)

if not torch.cuda.is_available():
    print("(torch reports no CUDA; skipping functional test)")
    sys.exit(0)

n = torch.cuda.device_count()
print(f"torch sees {n} CUDA device(s)\n")

failures = 0
for src in range(n):
    for dst in range(n):
        if src == dst:
            continue
        can = torch.cuda.can_device_access_peer(src, dst)
        try:
            a = torch.arange(1024, device=f"cuda:{src}", dtype=torch.float32)
            b = a.to(f"cuda:{dst}")
            ok = torch.equal(a.cpu(), b.cpu())
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"GPU {src} -> {dst}: can_access_peer={can}  copy ERROR: {e}")
            failures += 1
            continue
        status = "OK" if ok else "MISMATCH"
        if not ok:
            failures += 1
        print(f"GPU {src} -> {dst}: can_access_peer={can}  copy={status}")

print()
if failures:
    print(f"FAIL: {failures} GPU pair(s) failed the functional P2P copy.")
    sys.exit(1)
print("PASS: all GPU pairs can exchange data.")
PY
