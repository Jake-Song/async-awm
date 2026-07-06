#!/usr/bin/env bash
set -euo pipefail

apt update
apt install -y tmux python3-dev vim curl

echo 'set number' > ~/.vimrc

curl -LsSf https://astral.sh/uv/install.sh | sh

export UV_CACHE_DIR="/workspace/.cache/uv"
if ! grep -q 'UV_CACHE_DIR="/workspace/.cache/uv"' ~/.bashrc 2>/dev/null; then
    echo 'export UV_CACHE_DIR="/workspace/.cache/uv"' >> ~/.bashrc
fi
mkdir -p "$UV_CACHE_DIR"

# Size the OpenEnv MCP HTTP connection pool to the rollout concurrency. httpx's
# default pool caps total connections at 100, silently throttling concurrent
# rollouts; keep enough warm connections to avoid TCP-handshake churn between steps.
export OPENENV_MCP_MAX_KEEPALIVE_CONNECTIONS=512
if ! grep -q 'OPENENV_MCP_MAX_KEEPALIVE_CONNECTIONS' ~/.bashrc 2>/dev/null; then
    echo 'export OPENENV_MCP_MAX_KEEPALIVE_CONNECTIONS=512' >> ~/.bashrc
fi

cd ..
git clone https://github.com/Jake-Song/trl.git
cd trl
git checkout AsyncGRPO
cd ..
git clone https://github.com/Jake-Song/OpenEnv.git
cd OpenEnv
git checkout llm-judge-batch
cd ..
curl -fsSL https://claude.ai/install.sh | bash
