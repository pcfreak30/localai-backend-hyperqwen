"""Assemble the standalone Linux/glibc runtime from the pinned HQ image.

Requires a compatible glibc host (Ubuntu 24.04 baseline); not a rootfs/container.
Compiler and driver requirements are explicit, not silently borrowed at build time.
"""
import os
from pathlib import Path
import shutil

from toolchain import assemble

out = Path('/package')
shutil.copytree('/app', out / 'hq', symlinks=True)
python = out / 'python'
(python / 'bin').mkdir(parents=True)
shutil.copy2('/usr/bin/python3.12', python / 'bin/python3.12')
shutil.copytree('/usr/lib/python3.12', python / 'lib/python3.12', symlinks=True)
shutil.copytree('/usr/include/python3.12', python / 'include/python3.12', symlinks=True)
lib = out / 'lib'
lib.mkdir()
# Do not bundle glibc/loader: the initial package explicitly targets glibc >= 2.39.
for pattern in ('libexpat.so*', 'libz.so*', 'libstdc++.so*', 'libgcc_s.so*', 'libnuma.so*', 'libgomp.so*', 'libffi.so*', 'libssl.so*', 'libcrypto.so*', 'libsqlite3.so*', 'libbz2.so*', 'liblzma.so*', 'libreadline.so*', 'libtinfo.so*', 'libuuid.so*', 'libcrypt.so*'):
    for path in Path('/usr/lib/x86_64-linux-gnu').glob(pattern):
        shutil.copy2(path.resolve(), lib / path.name)
# Keep HQ's environment layout but replace absolute interpreter symlinks.
for name in ('python', 'python3', 'python3.12'):
    path = out / 'hq/venv/bin' / name
    path.unlink(missing_ok=True)
    path.symlink_to('../../../python/bin/python3.12')
for path in (out / 'hq/venv/bin').iterdir():
    if path.is_file() and not path.is_symlink():
        data = path.read_bytes()
        if data.startswith(b'#!/app/venv/bin/python'):
            path.write_bytes(b'#!/usr/bin/env python3\n' + data.split(b'\n', 1)[1])
# CUDA toolkit remains required for FlashInfer JIT. Its normal layout is relocatable.
shutil.copytree('/usr/local/cuda-13.0', out / 'cuda', symlinks=True)
shutil.copytree('/adapter-deps', out / 'adapter-deps')
for name in ('backend_pb2.py', 'backend_pb2_grpc.py'):
    shutil.copy2('/adapter/' + name, out / name)
shutil.copy2('/adapter/packaging/run.sh', out / 'run.sh')
shutil.copytree('/adapter/licenses', out / 'licenses')
for name in ('LICENSE', 'THIRD_PARTY_NOTICES.txt'):
    shutil.copy2('/adapter/' + name, out / name)
os.chmod(out / 'run.sh', 0o755)
assemble(out)
for name in ('runtime.lock.json', 'requirements.lock'):
    shutil.copy2('/adapter/' + name, out / name)
