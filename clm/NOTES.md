# CLM-8B as a fifth selector, run locally

CLM ("Contrastive Language Model", Stanford/NVIDIA, released 2026-09-23, Apache-2.0): a frozen Qwen3-8B
encoder plus two projection heads trained with InfoNCE. It scores each candidate action by the cosine
between its projected embedding and the projected state embedding. Package `contrastive-lm` 0.1.0 (PyPI,
sdist sha256 `2bfe2c57...`; identical inference code to github.com/Contrastive-LM/CLM at `d5f9ef0`),
reference head `Contrastive-LM/CLM-v0.1-8B` / `CLM_v0.1-8B.pt` (75.6 MB, sha256 `b2b4a8c9...`, matches the Hub).

## What CLM needs (read from the source)

- **Encoder**: an OpenAI-compatible `/v1/embeddings`, reference `vllm serve Qwen/Qwen3-8B --runner pooling`.
  `clm.embedder.Embedder` POSTs `{"model": "qwen3-8b", "input": [texts], "encoding_format": "base64",
  "truncate_prompt_tokens": 2048}` in batches of 32 and L2-normalises what comes back (so the server's own
  normalisation doesn't matter). vLLM's default pooler for a causal LM converted to embeddings is LAST
  (`as_embedding_model`: "the normalized hidden state corresponding to the last token"); string input is
  tokenised with `add_special_tokens=True`, which for Qwen3 adds nothing (tokenizer.json's post-processor
  is ByteLevel only, `add_bos_token: false`; the 0.6B and 8B tokenizer.json are byte-identical). That is
  also the training precompute (`train/embed_utils.py`: `add_special_tokens=False`, LAST pooling, l2).
  So the target is: token ids of the raw text, hidden state of the last token after the final RMSNorm,
  4096-d, L2-normalised. vLLM would run it in bf16 (Qwen3-8B's `torch_dtype`).
- **Heads**: `cfg = {width 1536, depth 3, projection_dim 512, gelu, layernorm}`, 9.44M parameters each,
  `logit_scale` 4.613 -> exp clamped to 100: logits are 100 x cosine. Run on the CPU here (torch CPU wheel).
- **Question format**: TypeSafe's System One wire format, so Jev's request body replays unchanged. For a
  `choice` question the state head reads `state + "\n\n" + instructions` (`clm.schema.state_text`) and the
  action head reads each option's **description only** (the key, here the tool name, is embedded only when
  the description is empty; none of the 91 are). Answer = softmax over the 91 logits; `confidence` = top
  probability minus the mean of the others (TypeSafe's definition). `usage.input_tokens` counts encoder
  tokens spent on cache misses only.

## How it is served here

The GTX 1080 Ti (sm_61, Pascal) can't run vLLM, so the encoder is **llama.cpp** (v0.7.0, `620d222`):

1. `build_llamacpp.sh`: llama-server built in `nvidia/cuda:12.4.1-devel-ubuntu22.04` with
   `-DCMAKE_CUDA_ARCHITECTURES=61` (real sm_61 code; the official CUDA images are CUDA 12.8 with sm_61 only as
   PTX, which driver 560.94 / CUDA 12.6 can't JIT). cmake from its PyPI wheel (apt mirrors unreachable).
2. `serve_encoder.sh`: `llama-server --embeddings --pooling last --fit off -ngl N -c 2048 -b 1024 -ub 1024
   -np 1 -t 6 -tb 6` in a container on 127.0.0.1:8090, model `Qwen/Qwen3-8B-GGUF` `Qwen3-8B-Q8_0.gguf`
   (official, sha256 `408b9555...`). Its tokenizer metadata (qwen2 pre-tokenizer, `add_bos_token = false`,
   no add_eos) is the same as the 0.6B GGUF that tokenises identically to HF below.
3. `serve_clm.sh`: `clm-serve --device cpu --no-ui` on 127.0.0.1:8700, the stock package, nothing patched.
4. `../scripts/run_clm.py` (toolpick env): Jev's final-setup request to `POST /v1/systemone`.

The environment (`pyproject.toml`, `uv.lock`): Python 3.12, `contrastive-lm==0.1.0` with its `vllm`
requirement removed by `override-dependencies` (and `environments = linux`, without which the universal lock
still opened a fork for vllm, backtracked to vllm 0.1.3 and pulled pydantic down to 1.x, which breaks
FastAPI), torch 2.14.1+cpu, fastapi, uvicorn, huggingface_hub; `validate` group: transformers, gguf,
sentencepiece for the parity check. Files outside the repo: `~/.cache/clm/` (head, GGUFs, llama.cpp source
and build, cmake).

## Validation: does the encoder reproduce the reference?

`parity_06b.py` computes the reference recipe with HF transformers in fp32 for **Qwen3-0.6B** (same tokenizer
and architecture; fits in RAM) on 12 real CLM inputs from the study (6 state texts: fresh message + routing
policy + question; 6 action texts from 6 to 266 tokens) and compares with llama-server's `/v1/embeddings`:

| encoder (Qwen3-0.6B) | token ids identical | cosine vs HF fp32 last-token: min / mean |
|---|---|---|
| llama.cpp, F32 GGUF converted from the HF weights, CPU | 12/12 | 1.000000 / 1.000000 |
| llama.cpp, official Q8_0 GGUF, GPU (all layers) | 12/12 | 0.999064 / 0.999394 |
| llama.cpp, official Q8_0 GGUF, CPU | 12/12 | 0.999018 / 0.999379 |
| *for scale:* HF bf16 (vLLM's default dtype) | (same ids) | 0.999870 / 0.999920 |
| *negative control:* HF mean pooling instead of last-token | | 0.31 (min) |
| *negative control:* HF with `<|endoftext|>` appended | | 0.03 (min) |

So tokenisation, pooling, the post-norm hidden state and normalisation match exactly (F32: cosine 1.000000);
the only remaining error is Q8_0 rounding, about 7x the bf16 rounding the reference itself carries. The 8B
reference (bf16, 16 GB) doesn't fit this machine, so the 8B Q8_0 encoder error is not measured directly.

**8B, end to end.** The 8B GGUF tokenises the 12 strings identically to HF (12/12) and is deterministic.
The README's playground screenshot (`assets/playground.png`, "captured against a real clm-serve, clm-latest,
Qwen3-8B encoder on one RTX 4090", i.e. the vLLM bf16 reference) shows one full request; replayed here:

| README example request | reference screenshot | this setup (Q8_0) |
|---|---|---|
| urgency (noul), p(true) | 84.8% | 84.6% |
| department (choice) | billing 98.8%, confidence 97.6% | billing 98.8%, confidence 97.6% |
| frustration (score) | 2.00, Very angry 100.0% | 2.00, Very angry 100.0% |
| `rank` tides example (README / model card) | 0.997 / 0.993 | 0.995 |

The 0.2-point noul gap is a logit difference of about 0.015 (a cosine gap of 0.00015). The README's prose
numbers for the same request (0.41 / 0.939 / 1.98, "106 tokens") come from an older template: the
current package sends 98 tokens for it, exactly what the HF tokenizer counts.

## Results (audited key, `uv run --no-sync python clm/score_clm.py`)

Fresh 600 (`runs/fresh/clm.jsonl`), Jev's final setup exactly (same options, same instructions):

| selector | correct | benign | harmful | hand-off | ask-first rule | median time (p95) |
|---|---|---|---|---|---|---|
| **CLM-8B (local, Q8_0)** | **35.3%** | 47.0% | **17.7%** | 46.0% | 32.3% | 1.07 s (1.15), pick only |
| Jev | 95.3% | 4.5% | 0.2% | 61.0% | 90.3% | 0.26 s pick; 1.39 s with the filling call |
| Claude Sonnet 5.5 | 93.3% | 6.7% | 0.0% | 52.0% | 93.0% | 1.79 s |
| Claude Haiku 5.5 | 82.2% | 17.8% | 0.0% | 41.7% | 81.8% | 1.65 s |
| DeepSeek V4.1 Flash | 47.2% | 52.5% | 0.3% | 7.5% | 47.0% | 2.02 s |
| *always hand off* | *59.2%* | | | | | |

Exact McNemar on the same 600: CLM vs Jev only-CLM-right 5, only-Jev-right 365, p = 5e-101 (difference
-60.0 points, bootstrap 95% CI -64.0 to -56.2); vs Sonnet 13 / 361, p = 2e-89 (-58.0, CI -62.3 to -53.7).
Ask-first rule: p = 4e-100 and 7e-95. CLM and Jev pick the same tool on 32.5% of messages.

First 600 (`runs/experiments/clm_first.jsonl`, final setup): CLM 42.2% correct, 48.7% benign, 9.2% harmful,
61.7% hand-off; Jev's final setup on the same messages 96.8% (1 / 329, p = 3e-97).

**Failure pattern.** CLM uses 10 of the 91 options (8 on the first sample): hand_off_to_human 276,
request_return 104, get_most_recent_order_status 88, get_order_status 56, search_catalog 56. Four of those
five carry the "Customers say: ..." lines, the text closest to a customer message whatever the intent. It
never picks the FAQ tool (Jev: 112 times), so every policy question fails: check_cancellation_fee,
delivery_options, check_payment_methods, delivery_period and check_refund_policy are 0/22-23 each
(hand-offs, or request_return for refund-policy questions). Harmful picks are mostly `request_return` (89: a
write nobody asked for) and `notion.API-get-user` (15) on account intents. Picks are close calls: median
top-1 vs top-2 logit margin 0.47 (a cosine gap of 0.005), median confidence 0.28 (harmful 0.24, correct
0.35). A confidence gate can't rescue it: handing off below 0.3 acts on 14% of messages, 30% of those
still harmful, 57.3% correct, which is below always handing off (59.2%).

**Diagnostics** (`run_clm.py --variant ... --out clm/cache/...`, fresh 600; exploratory, not in runs/):

| one change from the final setup | correct | benign | harmful | hand-off | note |
|---|---|---|---|---|---|
| none (final setup) | 35.3% | 47.0% | 17.7% | 46.0% | |
| option text = "tool_name: description" | 31.7% | 14.0% | 54.3% | 42.0% | request_return 341 times |
| descriptions without the six customer-word lines | 58.7% | 36.0% | 5.3% | 86.3% | hands off 518/600: the always-hand-off baseline |
| question without the 800-character policy | 49.8% | 43.3% | 6.8% | 72.5% | |
| no instructions (state = the message alone) | 6.0% | 27.0% | 67.0% | 4.2% | topical but wrong: create_invoice 83, list_payment_intents 69 |

Without instructions CLM matches topic (an invoice question -> create_invoice), not the right action; the
policy and question pull it towards hand-off and the customer-worded tools. No variant separates intents.

**Speed.** 1.07 s per message (server-side 1.066 s: all encoder; ~182 encoder tokens, the state), p95
1.15 s; 600 messages in about 11 minutes, sequential. One-time: the 91 descriptions (about 5,200 tokens)
took 89 s, then they are cached. llama-bench pp192 for this encoder: 99 tok/s at -ngl 0, 149 at 12, 178 at
16 (CPU-resident layers are streamed to the GPU over PCIe for batched matmuls). The reference setup
(vLLM, RTX 4090) quotes ~28 ms server-side per new state, so the latency here says nothing about CLM itself.

## Caveats

- **Quantisation.** Encoder is Qwen3-8B Q8_0 in llama.cpp (16 of 36 layers in VRAM), not bf16 in vLLM.
  Measured: Q8_0 vs fp32 cosine >= 0.9990 at 0.6B, and the 8B README request reproduced to display precision.
  Upper bound on the effect: 48 of 600 picks have a top-two logit margin under 0.05 (3x the gap seen on the
  README request); flipping all of them moves CLM between 33.7% and 36.3%. Even flipping every pick under a
  0.25 margin (195) gives 29.0%-40.7%. The conclusion doesn't depend on the quantisation.
- **Zero-shot, reference head.** The README's SOTA verifier numbers are fine-tuned heads; this is
  `clm-latest` as released. CLM embeds option descriptions, not names (by design: `clm.schema.candidates`).
- **Pick only.** No argument-filling call (Jev's 1.39 s and $0.34/1,000 include one); CLM costs $0 (local).
- **`toolpick` doesn't know CLM yet** (shared code not edited): `parse_run_name("fresh/clm")` returns
  `{'mode': 'clm', 'model': '', 'prompt': 'final', 'sample': 'fresh'}` and `experiments/clm_first` returns
  model "first", prompt "base". Needs a `clm` branch like the `jev` one (model "clm", prompt "final") and
  `MODEL_NAMES["clm"]`; `notebooks/explore.py`'s mode filter (jev/router/native) hides mode "clm", and
  `scripts/build_ledger.py`'s `run_meta` labels `experiments/clm_first` mode "router". Nothing crashes.

## Reproduce

```bash
# once: llama.cpp with CUDA for sm_61, the GGUF, the CLM env and head (files under ~/.cache/clm/)
git clone --depth 1 --branch v0.7.0 https://github.com/ggml-org/llama.cpp.git ~/.cache/clm/llama.cpp-src
pip download cmake==3.31.6 --only-binary :all: --no-deps -d /tmp/dl && python -m zipfile -e /tmp/dl/cmake-*.whl ~/.cache/clm/tools/cmake
clm/build_llamacpp.sh
curl -L -o ~/.cache/clm/gguf/Qwen3-8B-Q8_0.gguf https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q8_0.gguf
(cd clm && uv sync && uv run clm-download)      # head -> ~/.cache/clm/CLM_v0.1-8B.pt

# validation (0.6B): HF fp32 reference vs llama.cpp
curl -L -o ~/.cache/clm/gguf/Qwen3-0.6B-Q8_0.gguf https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf
MODEL=Qwen3-0.6B-Q8_0.gguf NGL=all PORT=8091 NAME=clm-enc06 clm/serve_encoder.sh
(cd clm && uv run python parity_06b.py --url http://127.0.0.1:8091 && uv run python parity_06b.py --bf16-control)
docker stop clm-enc06

# the run (nvidia-smi first: -ngl 16 needs ~4.1 GB of VRAM)
MODEL=Qwen3-8B-Q8_0.gguf NGL=16 clm/serve_encoder.sh
clm/serve_clm.sh &
uv run --no-sync python scripts/run_clm.py --sample fresh       # runs/fresh/clm.jsonl (resumes)
uv run --no-sync python scripts/run_clm.py --sample first       # runs/experiments/clm_first.jsonl
uv run --no-sync python clm/score_clm.py                        # or --run experiments/clm_first, --file clm/cache/clm_names.jsonl
docker stop clm-encoder                                         # and stop clm-serve
```
