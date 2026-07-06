#!/usr/bin/env bash
set -euo pipefail

uv venv
uv pip install -e ../trl[vllm]
uv pip install -U transformers
uv pip install wandb bitsandbytes kernels==0.14.0
uv pip install -e ../OpenEnv/envs/agent_world_model_env
uv pip install python-dotenv
# Pin below 0.137: its include_router refactor adds _IncludedRouter routes (no .path),
# which breaks prometheus-fastapi-instrumentator's metrics middleware on the vLLM server.
uv pip install "fastapi<0.137"
