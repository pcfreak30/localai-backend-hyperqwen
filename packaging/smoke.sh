#!/bin/sh
# Checks only: no image pulls, extraction, downloads or model/GPU use.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PACKAGE=$(CDPATH= cd -- "${1:?usage: smoke.sh EXTRACTED_DIRECTORY}" && pwd)
test -x "$PACKAGE/run.sh"
docker run --rm --pull=never --network none --read-only \
  --tmpfs /tmp:rw,exec,size=512m \
  -e HOME=/tmp -e XDG_CACHE_HOME=/tmp/.cache \
  -e PYTHONHOME=/opt/hq/python \
  -e PYTHONPATH=/opt/hq:/opt/hq/adapter-deps:/opt/hq/hq/venv/lib/python3.12/site-packages \
  -e LD_LIBRARY_PATH=/opt/hq/lib \
  -e CUDA_HOME=/opt/hq/cuda \
  -v "$PACKAGE:/opt/hq:ro" \
  -v "$ROOT/tests/package_probe.py:/checks/package_probe.py:ro" \
  ubuntu:24.04 /bin/sh -c '
    set -eu
    /opt/hq/python/bin/python3.12 /checks/package_probe.py /opt/hq
    # Native HQ must resolve its own dependencies, not adapter overrides.
    export PYTHONPATH=/opt/hq/hq/venv/lib/python3.12/site-packages
    export PATH=/opt/hq/hq/venv/bin:/opt/hq/cuda/bin:$PATH
    /opt/hq/python/bin/python3.12 -c "import grpc, torch, vllm; print(\"Native HQ imports OK:\", grpc.__version__, torch.__version__, vllm.__version__)"
    cd /opt/hq/hq
    export HOST=127.0.0.1 PRINT_ARGV=1 SPEC=off CTX=fast MODEL=/nonexistent
    bash single-user/start_qwen.sh
    bash batch/start_qwen.sh
    /opt/hq/cuda/bin/nvcc --version
  '
sh "$ROOT/packaging/compile-probe.sh" "$PACKAGE"
