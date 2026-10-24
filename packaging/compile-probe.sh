#!/bin/sh
# Compile with bundled tools in clean Ubuntu, without a host compiler or GPU.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PACKAGE=$(CDPATH= cd -- "${1:?usage: compile-probe.sh EXTRACTED_DIRECTORY}" && pwd)
test -x "$PACKAGE/toolchain/bin/gcc"
docker run --rm --pull=never --network none --read-only \
  --tmpfs /tmp:rw,exec,size=512m \
  -e HOME=/tmp -e XDG_CACHE_HOME=/tmp/.cache \
  -v "$PACKAGE:/opt/hq:ro" -v "$ROOT/packaging:/checks:ro" \
  ubuntu:24.04 /bin/sh -c '
    set -eu
    test ! -x /usr/bin/gcc
    export PATH=/opt/hq/toolchain/bin:/opt/hq/toolchain/usr/bin:$PATH
    export LD_LIBRARY_PATH=/opt/hq/lib
    export PYTHONHOME=/opt/hq/python
    g++ /checks/probe_host.cpp -o /tmp/probe
    /tmp/probe
    gcc -shared -fPIC -I/opt/hq/python/include/python3.12 /checks/probe_python.c -o /tmp/_hq_probe.so
    PYTHONPATH=/tmp /opt/hq/python/bin/python3.12 -c "import _hq_probe; assert _hq_probe.answer() == 42; print(\"Bundled Python extension OK\")"
    /opt/hq/cuda/bin/nvcc -arch=sm_86 -c /checks/probe_kernel.cu -o /tmp/probe.o
    test -s /tmp/probe.o
    echo "Bundled-GCC CUDA sm86 compilation OK (no host GCC)"
  '
