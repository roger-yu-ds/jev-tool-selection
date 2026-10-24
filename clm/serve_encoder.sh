#!/bin/bash
# CLM's encoder on this machine: llama-server (built by build_llamacpp.sh) serving a Qwen3 GGUF through
# the OpenAI-compatible /v1/embeddings that CLM's Embedder calls. This stands in for the reference
#   vllm serve Qwen/Qwen3-8B --served-model-name qwen3-8b --runner pooling --max-model-len 2048
# (vLLM does not support the GTX 1080 Ti). Same recipe: raw-text token ids (Qwen3 adds no BOS/EOS),
# hidden state of the last token after the final RMSNorm (--pooling last), L2-normalised.
#
#   MODEL=Qwen3-8B-Q8_0.gguf NGL=16 ./serve_encoder.sh          # the run: port 8090, 16 of 36 layers on the GPU
#   MODEL=Qwen3-0.6B-Q8_0.gguf NGL=all PORT=8091 NAME=clm-enc06 ./serve_encoder.sh    # the parity check
#   docker logs -f clm-encoder; docker stop clm-encoder
#
# VRAM: the GPU is shared with always-on services (~6 GB); check nvidia-smi first and leave >= 500 MB free.
# --fit off so the layer count is exactly NGL. Threads capped at 6 (6 cores / 12 threads).
set -euo pipefail
MODEL="${MODEL:-Qwen3-8B-Q8_0.gguf}"
NGL="${NGL:-16}"
PORT="${PORT:-8090}"
THREADS="${THREADS:-6}"
NAME="${NAME:-clm-encoder}"
GGUF_DIR="${GGUF_DIR:-$HOME/.cache/clm/gguf}"
BIN_DIR="${BIN_DIR:-$HOME/.cache/clm/llama.cpp-src/build-cuda61/bin}"
IMAGE="${IMAGE:-nvidia/cuda:12.4.1-devel-ubuntu22.04}"
GPUS=(--gpus all)   # also needed for NGL=0: the binary links libcuda, which the nvidia runtime mounts
[ "$NGL" = "0" ] && GPUS+=(-e CUDA_VISIBLE_DEVICES=)   # CPU only: hide the GPU, allocate no VRAM

docker run -d --rm --name "$NAME" "${GPUS[@]}" --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -p "127.0.0.1:$PORT:8080" -v "$BIN_DIR":/opt/llama:ro -v "$GGUF_DIR":/models:ro "$IMAGE" \
  /opt/llama/llama-server -m "/models/$MODEL" --host 0.0.0.0 --port 8080 \
  --embeddings --pooling last --fit off -ngl "$NGL" -c 2048 -b 1024 -ub 1024 -np 1 \
  -t "$THREADS" -tb "$THREADS" --no-webui
echo "started $NAME on 127.0.0.1:$PORT ($MODEL, -ngl $NGL); waiting for /health"
for _ in $(seq 1 180); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { echo "ready"; exit 0; }
  docker ps -q -f "name=^${NAME}$" | grep -q . || { echo "container exited"; exit 1; }
  sleep 2
done
echo "not ready after 6 min; see: docker logs $NAME"; exit 1
