"""Real subprocess crash tests; only synthetic CPU fixtures are launched."""
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest

import grpc
import backend_pb2 as pb
import backend_pb2_grpc as rpc
from tests.checkpoint_fixture import create_checkpoint


def running(pid):
    try:
        # An adopted zombie no longer runs/holds GPU resources; init reaps it.
        return Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


def wait_until(predicate, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError('subprocess condition timed out')


@unittest.skipUnless(sys.platform == 'linux', 'Linux process-group cleanup')
class CrashCleanupTests(unittest.TestCase):
    def test_adapter_sigkill_removes_term_resistant_workers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            worker = root / 'worker.py'
            worker.write_text('import signal,time\nsignal.signal(signal.SIGTERM,signal.SIG_IGN)\ntime.sleep(60)\n')
            launcher = root / 'launcher.py'
            launcher.write_text('import os,signal,subprocess,sys,time\nfrom pathlib import Path\n'
                                'signal.signal(signal.SIGTERM,signal.SIG_IGN)\n'
                                f'p=subprocess.Popen([sys.executable,{str(worker)!r}])\n'
                                f'Path({str(root / "workers")!r}).write_text(str(os.getpid())+" "+str(p.pid))\n'
                                'time.sleep(60)\n')
            parent_code = ('import os,sys,time\nfrom pathlib import Path\n'
                           'from hyperqwen_backend.runtime import Runtime\nr=Runtime()\n'
                           f'r.start([sys.executable,{str(launcher)!r}],dict(os.environ),{str(root)!r})\n'
                           f'Path({str(root / "guardian")!r}).write_text(str(r.process.pid))\n'
                           'time.sleep(60)\n')
            parent = subprocess.Popen([sys.executable, '-c', parent_code])
            guardian = None
            try:
                wait_until(lambda: (root / 'workers').exists() and (root / 'guardian').exists())
                guardian = int((root / 'guardian').read_text())
                workers = [int(pid) for pid in (root / 'workers').read_text().split()]
                parent.kill()
                parent.wait(timeout=5)
                wait_until(lambda: not any(running(pid) for pid in [guardian] + workers))
            finally:
                parent.kill() if parent.poll() is None else None
                parent.wait()
                if guardian is not None:
                    try:
                        os.killpg(guardian, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    def test_controller_sigkill_stops_adapter_and_native_child(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / 'model'
            model.mkdir()
            create_checkpoint(model)
            launcher = root / 'single-user/start_qwen.sh'
            launcher.parent.mkdir()
            fake = Path(__file__).with_name('fake_hq.py').resolve()
            launcher.write_text(f'echo $$ > {shlex.quote(str(root / "native-pid"))}\n'
                                f'exec {shlex.quote(sys.executable)} {shlex.quote(str(fake))}\n')
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0))
                port = sock.getsockname()[1]
            code = ('import subprocess,sys,time\nfrom pathlib import Path\n'
                    f'p=subprocess.Popen([sys.executable,"-m","hyperqwen_backend.server","--addr","127.0.0.1:{port}"])\n'
                    f'Path({str(root / "adapter-pid")!r}).write_text(str(p.pid))\n'
                    'time.sleep(60)\n')
            env = dict(os.environ, HQ_ROOT=str(root), HQ_STARTUP_TIMEOUT='5')
            controller = subprocess.Popen([sys.executable, '-c', code], env=env, start_new_session=True)
            channel = grpc.insecure_channel(f'127.0.0.1:{port}')
            try:
                grpc.channel_ready_future(channel).result(timeout=10)
                result = rpc.BackendStub(channel).LoadModel(pb.ModelOptions(Model=str(model)), timeout=10)
                self.assertTrue(result.success, result.message)
                adapter = int((root / 'adapter-pid').read_text())
                native = int((root / 'native-pid').read_text())
                controller.kill()
                controller.wait(timeout=5)
                wait_until(lambda: not running(adapter) and not running(native))
            finally:
                channel.close()
                if controller.poll() is None:
                    controller.kill()
                controller.wait()
                try:
                    os.killpg(controller.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
