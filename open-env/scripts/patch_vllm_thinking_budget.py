"""Patch the installed vLLM so /v1/completions accepts `thinking_token_budget`.

vLLM (v0.19.x) only exposes thinking_token_budget on /v1/chat/completions, but
the async-GRPO rollout worker generates via /v1/completions. This adds the field
to CompletionRequest and passes it through to SamplingParams (which already
supports it). Idempotent; run_vllm_awm.sh runs it before `vllm serve`.
"""

import vllm.entrypoints.openai.completion.protocol as protocol

FIELD_ANCHOR = "    # --8<-- [end:completion-sampling-params]\n"
FIELD = "    thinking_token_budget: int | None = None\n"
PARAM_ANCHOR = "            prompt_logprobs=prompt_logprobs,\n"
PARAM = "            thinking_token_budget=self.thinking_token_budget,\n"

path = protocol.__file__
with open(path) as f:
    src = f.read()

if "thinking_token_budget" in src:
    print(f"already patched: {path}")
else:
    if src.count(FIELD_ANCHOR) != 1 or src.count(PARAM_ANCHOR) != 1:
        raise SystemExit(
            f"anchors not found in {path} — vLLM layout changed, patch manually"
        )
    src = src.replace(FIELD_ANCHOR, FIELD + FIELD_ANCHOR)
    src = src.replace(PARAM_ANCHOR, PARAM_ANCHOR + PARAM)
    with open(path, "w") as f:
        f.write(src)
    print(f"patched: {path}")
