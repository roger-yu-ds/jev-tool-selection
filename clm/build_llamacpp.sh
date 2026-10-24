#!/bin/bash
# Build llama.cpp (llama-server, llama-bench) with CUDA for the
# GTX 1080 Ti (sm_61).
#
# Why a container: there is no CUDA toolkit on the host, and the official llama.cpp CUDA images are
# built with CUDA 12.8 and ship sm_61 only as PTX, which the host driver (560.94, CUDA 12.6) cannot JIT.
# CUDA 12.4.1 is supported by that driver, so build real sm_61 code with it. The devel image has gcc,
# make and nvcc but no cmake, and apt mirrors are unreachable from here, so cmake comes from its PyPI
# wheel (cmake-3.31.6 manylinux2014), unpacked into $CMAKE_DIR and mounted into the container.
#
#   git clone --depth 1 --branch v0.7.0 https://github.com/ggml-org/llama.cpp.git ~/.cache/clm/llama.cpp-src
#   pip download cmake==3.31.6 --only-binary :all: --no-deps -d dl && python -m zipfile -e dl/cmake-*.whl ~/.cache/clm/tools/cmake
#   ./build_llamacpp.sh        # ~15-25 min on 6 threads
set -euo pipefail
LLAMA_SRC="${LLAMA_SRC:-$HOME/.cache/clm/llama.cpp-src}"
CMAKE_DIR="${CMAKE_DIR:-$HOME/.cache/clm/tools/cmake/cmake/data}"
JOBS="${JOBS:-6}"
IMAGE="${IMAGE:-nvidia/cuda:12.4.1-devel-ubuntu22.04}"

docker run --rm --name clm-llamacpp-build --user "$(id -u):$(id -g)" \
  -v "$LLAMA_SRC":/src -v "$CMAKE_DIR":/opt/cmake:ro -w /src "$IMAGE" bash -c "
  set -e
  export PATH=/opt/cmake/bin:\$PATH HOME=/tmp
  cmake -B build-cuda61 -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61 -DGGML_NATIVE=ON \
        -DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_OPENSSL=OFF
  cmake --build build-cuda61 -j $JOBS --target llama-server llama-bench
"
ls -la "$LLAMA_SRC/build-cuda61/bin"
