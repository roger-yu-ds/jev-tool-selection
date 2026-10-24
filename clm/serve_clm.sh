#!/bin/bash
# CLM's System One API: clm-serve (contrastive-lm 0.1.0) with the reference head CLM_v0.1-8B.pt
# (sha256 b2b4a8c9...; `uv run clm-download` fetches it into ~/.cache/clm/), heads on the CPU,
# encoder = llama-server from serve_encoder.sh. Listens on 127.0.0.1:8700 only.
#
#   ./serve_clm.sh            # then: curl -s 127.0.0.1:8700/health
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"   # the heads are tiny; leave the cores to llama.cpp
exec uv run --no-sync clm-serve --host 127.0.0.1 --port "${PORT:-8700}" \
  --emb-url "${EMB_URL:-http://127.0.0.1:8090/v1/embeddings}" --emb-model qwen3-8b \
  --ckpt "${CLM_CKPT:-$HOME/.cache/clm/CLM_v0.1-8B.pt}" --no-download --device cpu --no-ui
