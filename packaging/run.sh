#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
# HQ_ROOT is a trusted operator override (also used by CPU integration fixtures).
export HQ_ROOT="${HQ_ROOT:-$ROOT/hq}"
export PYTHONHOME="$ROOT/python"
export HQ_NATIVE_PYTHONPATH="$ROOT/hq/venv/lib/python3.12/site-packages"
export PYTHONPATH="$ROOT:$ROOT/adapter-deps:$HQ_NATIVE_PYTHONPATH"
export PATH="$ROOT/hq/venv/bin:$ROOT/toolchain/bin:$ROOT/toolchain/usr/bin:$ROOT/cuda/bin:$PATH"
export CC="${CC:-$ROOT/toolchain/bin/gcc}"
export CXX="${CXX:-$ROOT/toolchain/bin/g++}"
export CUDAHOSTCXX="${CUDAHOSTCXX:-$ROOT/toolchain/bin/g++}"
export LD_LIBRARY_PATH="$ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_HOME="$ROOT/cuda"
export CUDA_PATH="$ROOT/cuda"
# Preserve caller-owned HOME/cache placement. Never create or modify model files.
exec "$ROOT/python/bin/python3.12" -m hyperqwen_backend.server "$@"
