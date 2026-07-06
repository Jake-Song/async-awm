#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CUDA_VISIBLE_DEVICES=6,7 \
    uv run accelerate launch \
        --config_file "$REPO_ROOT/open-env/configs/fsdp2.yaml" \
        "$REPO_ROOT/open-env/openenv_awm_async_grpo.py" "$@"
