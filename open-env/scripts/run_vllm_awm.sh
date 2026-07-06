#!/usr/bin/env bash
set -euo pipefail

if [ -f .env ]; then
  set -a && source .env && set +a
fi

uv run hf auth login --token "${HF_TOKEN}"

# /v1/completions doesn't accept thinking_token_budget upstream; patch it in.
uv run python "$(dirname "$0")/patch_vllm_thinking_budget.py"

# Concurrency limits (override via .env / environment). KV cache sits ~half-used
# at the defaults because concurrency is capped by count/token-budget, not memory.
# max_num_seqs caps the running batch by count; max_num_batched_tokens caps
# tokens/step (prefill + decode) and gates how fast the batch fills with long
# multi-turn prompts — raise both, then watch for KV preemptions as the real limit.
MAX_NUM_SEQS="${MAX_NUM_SEQS:-512}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"

# 8-GPU node split: 6 GPUs here for vLLM, 2 (GPUs 6,7) for the FSDP2 trainer.
# The model has 32 attention heads, so TP must divide 32 — plain TP=6 is invalid.
# Factor 6 = TP(2) x PP(3): TP=2 keeps 16 q-heads / 4 kv-heads per GPU, and the
# 36 decoder layers split evenly into 3 pipeline stages of 12.
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 VLLM_SERVER_DEV_MODE=1 \
    uv run vllm serve Jakemu/Qwen3-4B-Thinking-awm-async-grpo-100 \
        --tensor-parallel-size 2 \
        --pipeline-parallel-size 3 \
        --max-model-len 32768 \
        --max-num-seqs "${MAX_NUM_SEQS}" \
        --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
        --logprobs-mode processed_logprobs \
        --reasoning-parser deepseek_r1 \
        --reasoning-config '{"reasoning_start_str": "<think>", "reasoning_end_str": "</think>"}' \
        --weight-transfer-config '{"backend":"nccl"}' \
        --override-generation-config '{"temperature":1}' \
               
        # Qwen3 thinking-mode recommended sampling. The rollout request omits
        # top_p/top_k/min_p/presence_penalty so these server defaults apply; it
        # always sends temperature, so during training the trainer's
        # AsyncGRPOConfig.temperature overrides the 0.6 here (set there too).
