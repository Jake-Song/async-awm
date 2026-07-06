# AWM Async GRPO

How to run [`awm/async_grpo_awm.py`](awm/async_grpo_awm.py) —
async GRPO training for the Agent World Model (AWM) multi-turn MCP agent.

AWM is agentic: the model discovers MCP tools, calls them over several turns, and
a verifier scores the final outcome (`complete=1.0`, `incomplete=0.1`,
`format_error=-1.0`). Training uses TRL's `AsyncGRPOTrainer` with an
`environment_factory`, and a custom `AWMRolloutWorker` that scores each rollout
out-of-band (the reward is never shown to the model).

## Repo layout

```
.
├── awm/
│   ├── async_grpo_awm.py            # main async-GRPO training script
│   ├── config.yaml                  # default run config (CLI flags override)
│   ├── configs/
│   │   └── fsdp2.yaml               # accelerate FSDP2 config (6 sampling server and 2 trainer GPUs)
│   ├── scripts/
│   │   ├── install_packages.sh
│   │   ├── run_vllm_awm.sh          # vLLM server (GPUs 0–5, TP=2×PP=3)
│   │   ├── run_trainer_awm.sh       # FSDP2 trainer (GPUs 6,7)
│   │   ├── patch_vllm_thinking_budget.py
│   │   ├── check_judge_reliability.py
│   │   ├── list_dataset_groups.py
│   │   ├── simulate_gpu_allocation.py # simulation for optimal gpu allocation between sampling server and trainer
│   │   └── simulate_judge_cost.py
│   └── examples/
│       ├── awm_simple.py
│       └── awm_llm_judge.py
├── experiment/
│   ├── bfcl_to_awm.py               
│   ├── bfcl_data_examples.jsonl
│   ├── bfcl_data_examples.csv
│   └── LOG.md                       # analysis of rollouts and calibration
├── utils/
│   ├── check_gpu_p2p.sh             # check if peer to peer communication of GPUs
│   └── pretty_jsonl.py
├── startup.sh                       # provision a fresh cloud node
├── .env.example
├── pyproject.toml
├── CLAUDE.md
└── README.md
```

## Topology

An 8-GPU node, three processes:

| Process        | Where          | Brings up                                              |
|----------------|----------------|--------------------------------------------------------|
| AWM env server | CPU            | The MCP tools + verifier (FastAPI/uvicorn), port 8899  |
| vLLM server    | GPUs 0–5       | Generation backend, port 8000, NCCL weight transfer    |
| Trainer        | GPUs 6,7       | FSDP2 GRPO trainer; rollout worker runs on rank 0      |

vLLM runs `TP=2 × PP=3` across GPUs 0–5 (plain `TP=6` is invalid — the model's 32
attention heads aren't divisible by 6). The rollout worker only runs on rank 0,
so both trainer ranks share the single vLLM server. Rollouts hit the env server
synchronously over multiple turns, so throughput is bottlenecked by the env, not
the GPU.

## Prerequisites

- A custom TRL fork at `../trl` (sibling of this repo), branch `AsyncGRPO`. The
  async trainer's `importance_sampling_level` / `loss_type` knobs and
  `model_init_kwargs` live only in that fork — stock PyPI `trl` will fail with
  `unexpected keyword argument`.
- The AWM environment package from `../OpenEnv`, branch `llm-judge-batch`.
- An external LLM judge for the `sql` verifier (an OpenAI-compatible endpoint).
- An 8-GPU node (bf16 FSDP2 trainer). [`startup.sh`](startup.sh) provisions a
  fresh cloud node: installs `uv`, clones both repos on the right branches, and
  sets the MCP connection-pool env var.

## 1. Install

From the repo root:

```bash
bash awm/scripts/install_packages.sh
```

That runs (see the script for exact pins):

```bash
uv venv
uv pip install -e ../trl[vllm]                      # the custom TRL fork
uv pip install -U transformers
uv pip install wandb bitsandbytes kernels==0.14.0
uv pip install -e ../OpenEnv/envs/agent_world_model_env
uv pip install python-dotenv
uv pip install "fastapi<0.137"                       # >=0.137 breaks the vLLM metrics middleware
```

Log into Hugging Face and confirm the fork is the one in use:

```bash
uvx hf auth login
uv run python -c "import trl, os; print(trl.__version__, os.path.realpath(trl.__file__))"
# -> 1.6.0.dev0 .../trl/trl/__init__.py   (NOT site-packages)
```

## 2. Configure the ENV

The `sql` verifier calls an external LLM-as-judge. The env's `reset()` reads
these three variables, so export them (or put them in a `.env` file — the script
calls `load_dotenv()`):

```bash
export HF_TOKEN=
export WANDB_API_KEY=            #if you want wandb logging
export OPENENV_AWM_LLM_BASE_URL=https://your-openai-compatible-endpoint/v1
export OPENENV_AWM_LLM_API_KEY=sk-...
export OPENENV_AWM_LLM_MODEL=your-judge-model
```

(`OPENENV_AWM_LLM_BASE_URL` / `_MODEL` fall back to the `verifier:` section of
`config.yaml` if unset; the API key must come from the env.)

`huggingface_hub.login()` and `wandb.login()` run at startup. Set `HF_TOKEN` and
`WANDB_API_KEY` to avoid interactive prompts (or pass `--no-push-to-hub` to skip
the Hub upload).

## 3. Run (three terminals)

**Terminal 1 — AWM env server** (from the `OpenEnv` repo, on CPU):

```bash
cd ../OpenEnv
ulimit -n 65536
PYTHONPATH=src:envs uv run uvicorn \
  envs.agent_world_model_env.server.app:app --host 0.0.0.0 --port 8899 \
  --ws-ping-interval 1800 --ws-ping-timeout 1800
```

**Terminal 2 — vLLM server** (GPUs 0–5):

```bash
bash awm/scripts/run_vllm_awm.sh
```

**Terminal 3 — trainer** (GPUs 6,7, FSDP2 — the default):

```bash
export OPENENV_AWM_LLM_BASE_URL=... OPENENV_AWM_LLM_API_KEY=... OPENENV_AWM_LLM_MODEL=...
bash awm/scripts/run_trainer_awm.sh --env-url http://localhost:8899
```

`run_trainer_awm.sh` launches with `configs/fsdp2.yaml` (`num_processes: 2`,
wraps `Qwen3DecoderLayer`) on `CUDA_VISIBLE_DEVICES=6,7`. Any extra args are
forwarded to the script (see [Common flags](#common-flags)).

### Single trainer GPU

A single trainer GPU still works — launch `accelerate` directly instead of the
script:

```bash
CUDA_VISIBLE_DEVICES=1 uv run accelerate launch \
  awm/async_grpo_awm.py --env-url http://localhost:8899
```

## Configuration

Defaults live in [`awm/config.yaml`](awm/config.yaml) (loaded into the
`CONFIG` dict at startup); CLI flags override them. Edit the YAML for durable
changes, pass flags for one-offs.

### Common flags

The ones you'll touch most:

| Flag                        | Default                                       | Notes                              |
|-----------------------------|-----------------------------------------------|------------------------------------|
| `--model-id`                | `Jakemu/Qwen3-4B-Thinking-awm-async-grpo-100` | Policy + vLLM model (a checkpoint) |
| `--env-url`                 | `http://localhost:8899`                       | AWM env server                     |
| `--num-generations`         | `8`                                           | Rollouts per prompt (group size)   |
| `--max-turns`               | `20`                                          | Max tool-calling iterations        |
| `--max-completion-length`   | `4096`                                        | Tokens per completion              |
| `--gradient-accumulation-steps` | `16`                                      |                                    |
| `--learning-rate`           | `7e-7`                                         |                                    |
| `--save-steps`              | `10`                                           |                                    |
| `--no-push-to-hub`          | (push is on by default)                       | Skip the Hub upload                |
| `--output-dir`              | `Qwen3-4B-Thinking-awm-async-grpo-200`        | Local + Hub repo                   |
| `--wandb-project` / `--wandb-name` | `openenv-awm-continuous` / `awm-continuous-async-grpo-200` |            |


