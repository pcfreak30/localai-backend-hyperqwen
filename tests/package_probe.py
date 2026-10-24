"""Manual extracted-package import/service check, run inside clean Ubuntu."""
import signal
import socket
import subprocess
import sys
from pathlib import Path

import grpc
import httpx
import torch
import vllm
import backend_pb2 as pb
import backend_pb2_grpc as rpc
# Exercise the stdlib native libraries that previously needed bundled fixes.
import bz2, ctypes, lzma, readline, sqlite3, ssl  # noqa: F401

root = Path(sys.argv[1])
print('Runtime imports OK:', sys.version.split()[0], torch.__version__, vllm.__version__, flush=True)
with socket.socket() as sock:
    sock.bind(('127.0.0.1', 0))
    address = f'127.0.0.1:{sock.getsockname()[1]}'
process = subprocess.Popen([str(root / 'run.sh'), '--addr', address])
try:
    with grpc.insecure_channel(address) as channel:
        grpc.channel_ready_future(channel).result(timeout=30)
        stub = rpc.BackendStub(channel)
        assert stub.Health(pb.HealthMessage(), timeout=5).message == b'OK'
        result = stub.LoadModel(pb.ModelOptions(Model='/nonexistent-checkpoint'), timeout=5)
        assert not result.success and 'does not exist' in result.message, result
        assert stub.Free(pb.HealthMessage(), timeout=5).success
    process.send_signal(signal.SIGTERM)
    assert process.wait(timeout=15) == 0
    print('Packaged gRPC health / rejected load / free / shutdown OK', flush=True)
finally:
    if process.poll() is None:
        process.kill()
        process.wait()
