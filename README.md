# AWM Async GRPO

How to run [`openenv_awm_async_grpo.py`](openenv_awm_async_grpo.py) — async GRPO
training for the Agent World Model (AWM) multi-turn MCP agent.

AWM is agentic: the model discovers MCP tools, calls them over several turns, and
a verifier scores the final outcome. Training uses TRL's `AsyncGRPOTrainer` with
an `environment_factory`, and a custom `AWMRolloutWorker` that scores each rollout
out-of-band (the reward is never shown to the model).

## Topology

Three processes, each on its own resource:

| Process       | Where        | Brings up                                            |
|---------------|--------------|------------------------------------------------------|
| AWM env server| CPU          | The MCP tools + verifier (FastAPI/uvicorn), port 8899|
| vLLM server   | GPU 0        | Generation backend, port 8000, NCCL weight transfer  |
| Trainer       | GPU 1 (or 1–7)| FSDP2 GRPO trainer; rollout worker runs on rank 0   |

The rollout worker only runs on rank 0, so all trainer ranks share the single
vLLM server. Rollouts hit the env server synchronously over multiple turns, so
throughput is bottlenecked by the env, not the GPU.

## Prerequisites

- A custom TRL fork at `../trl` (sibling of this repo). The async trainer's
  `importance_sampling_level` / `loss_type` knobs and `model_init_kwargs` live
  only in that fork — stock PyPI `trl` will fail with
  `unexpected keyword argument`.
- The AWM environment package from `../OpenEnv`.
- An external LLM judge for the `sql` verifier (an OpenAI-compatible endpoint).
- GPUs with FP8 tensor cores (Hopper/Ada/Blackwell) for the default FP8 trainer
  config.
- **OOD eval only** (`experiment/mcp_universe_eval.py`): MCP-Universe
  **editable-installed from a clone** (`uv pip install -e ../MCP-Universe`, not a
  plain wheel — its BenchmarkRunner resolves task-JSON paths against the installed
  package dir), plus its live MCP servers and per-domain API keys
  (finance/yfinance, `GOOGLE_MAPS_API_KEY`). Not needed for training.

## 1. Install

From this `open-env/` directory:

```bash
bash scripts/install_packages.sh
```

That runs (see the script for exact pins):

```bash
uvx hf auth login                                   # Hugging Face login
uv venv
uv pip install -e ../trl[vllm]                      # the custom TRL fork
uv pip install -U transformers
uv pip install wandb bitsandbytes kernels==0.14.0
uv pip install -e ../OpenEnv/envs/agent_world_model_env
uv pip install python-dotenv
```

Confirm the fork is the one in use:

```bash
uv run python -c "import trl, os; print(trl.__version__, os.path.realpath(trl.__file__))"
# -> 1.6.0.dev0 .../trl/trl/__init__.py   (NOT site-packages)
```

## 2. Configure the LLM judge

The `sql` verifier calls an external LLM-as-judge. The env's `reset()` reads
these three variables, so export them (or put them in a `.env` file — the script
calls `load_dotenv()`):

```bash
export OPENENV_AWM_LLM_BASE_URL=https://your-openai-compatible-endpoint/v1
export OPENENV_AWM_LLM_API_KEY=sk-...
export OPENENV_AWM_LLM_MODEL=your-judge-model
```

`huggingface_hub.login()` and `wandb.login()` run at startup. Set `HF_TOKEN` and
`WANDB_API_KEY` to avoid interactive prompts (or pass `--no-push-to-hub` to skip
the Hub upload).

## 3. Run (three terminals)

**Terminal 1 — AWM env server** (from the `OpenEnv` repo, on CPU):

```bash
cd ../OpenEnv
PYTHONPATH=src:envs uv run uvicorn \
  envs.agent_world_model_env.server.app:app --host 0.0.0.0 --port 8899
```

**Terminal 2 — vLLM server** (GPU 0):

```bash
bash scripts/run_vllm_awm.sh
```

**Terminal 3 — trainer** (GPU 1, single-GPU FP8 — the default):

```bash
export OPENENV_AWM_LLM_BASE_URL=... OPENENV_AWM_LLM_API_KEY=... OPENENV_AWM_LLM_MODEL=...
bash scripts/run_trainer_awm.sh --env-url http://localhost:8899
```

`run_trainer_awm.sh` launches with `configs/accelerate_fp8_single_gpu.yaml` on
`CUDA_VISIBLE_DEVICES=1`. Any extra args are forwarded to the script (see
[Common flags](#common-flags)).

### Multi-GPU (7 trainer GPUs, FSDP2)

To shard the trainer across GPUs 1–7 (8 GPUs total with vLLM on GPU 0), launch
with the FSDP2 config instead of the single-GPU script:

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 \
  uv run accelerate launch \
    --config_file configs/fsdp2.yaml \
    openenv_awm_async_grpo.py --env-url http://localhost:8899
```

`configs/fsdp2.yaml` sets `num_processes: 7` and wraps `Qwen3DecoderLayer`.

## Common flags

Defaults live in `parse_args()`. The ones you'll touch most:

| Flag                        | Default                          | Notes                                |
|-----------------------------|----------------------------------|--------------------------------------|
| `--model-id`                | `Qwen/Qwen3-4B-Thinking-2507`    | Policy + vLLM model                  |
| `--env-url`                 | `http://localhost:8899`          | AWM env server                       |
| `--num-generations`         | `8`                              | Rollouts per prompt (group size)     |
| `--max-turns`               | `20`                             | Max tool-calling iterations          |
| `--max-completion-length`   | `2048`                           | Tokens per completion                |
| `--gradient-accumulation-steps` | `16`                         |                                      |
| `--learning-rate`           | `7e-7`                           |                                      |
| `--save-steps`              | `10`                             |                                      |
| `--no-push-to-hub`          | (push is on by default)          | Skip the Hub upload                  |
| `--output-dir`              | `Qwen/...-awm-async-grpo`        | Local + Hub repo                     |
| `--wandb-project` / `--wandb-name` | `openenv-awm` / `awm-async-grpo` |                              |

## Loss configuration

The run uses **sequence-level importance sampling (GSPO)** with `loss_type="grpo"`
and **no KL penalty** (`beta=0`). The async trainer has no reference model, and
`old_log_probs` are vLLM *sampling* logprobs rather than reference logprobs, so a
faithful KL term isn't available — see the comment in the config block. These
knobs require the `../trl` fork.

## Known issues

- `model_init_kwargs={"attn_implementation": "flash-attention_3"}` (config block,
  ~line 385) is likely a typo — HF expects `flash_attention_2`,
  `flash_attention_3`, or a `kernels-community/...` id. The hyphenated value can
  raise at model load. Adjust to match your installed attention kernels.

## Troubleshooting

- **`AsyncGRPOConfig got an unexpected keyword argument ...`** — the venv is using
  stock PyPI `trl`, not the fork. Re-run `uv pip install -e ../trl[vllm]` and
  recheck `trl.__file__`.
- **vLLM-ready / empty-queue timeouts** — multi-turn env rollouts are slow; bump
  `--vllm-server-timeout` (default 1200s).
- **Judge `TimeoutError` during scoring** — verify `OPENENV_AWM_LLM_*` point at a
  reachable endpoint; scoring failures are caught and scored `0.0`.
