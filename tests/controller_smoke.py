"""Opt-in pinned LocalAI API check with synthetic CPU fixtures; no downloads."""
import argparse
import hashlib
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
from urllib.error import URLError
from urllib.request import Request, ProxyHandler, build_opener

from tests.checkpoint_fixture import create_checkpoint


def main(binary, package=None):
    root = Path(__file__).resolve().parents[1]
    binary = Path(binary).resolve()
    pin = json.loads((root / 'runtime.lock.json').read_text())['tested_controller']
    with binary.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != pin['sha256']:
            raise ValueError('controller checksum does not match runtime.lock.json')
    (root / 'build').mkdir(exist_ok=True)
    log_path = root / 'build' / f'controller-{time.time_ns()}.log'
    print(f'Controller log: {log_path}', flush=True)
    with tempfile.TemporaryDirectory(prefix='hq-controller-') as directory:
        work = Path(directory)
        models = work / 'models'
        model = models / 'prepared-model'
        model.mkdir(parents=True)
        create_checkpoint(model)
        launcher = work / 'hq/single-user/start_qwen.sh'
        launcher.parent.mkdir(parents=True)
        launcher.write_text(f'exec {shlex.quote(sys.executable)} {shlex.quote(str(root / "tests/fake_hq.py"))}\n')
        wrapper = work / 'backends/hyperqwen/run.sh'
        wrapper.parent.mkdir(parents=True)
        if package:
            executable = Path(package).resolve() / 'run.sh'
            if not executable.is_file():
                raise ValueError(f'missing package launcher: {executable}')
            command = f'exec {shlex.quote(str(executable))} "$@"'
        else:
            command = f'export PYTHONPATH={shlex.quote(str(root))}\nexec {shlex.quote(sys.executable)} -m hyperqwen_backend.server "$@"'
        wrapper.write_text('#!/bin/sh\n' + command + '\n')
        wrapper.chmod(0o755)
        (models / 'probe.yaml').write_text(json.dumps({
            'name': 'hq-probe', 'backend': 'hyperqwen',
            'parameters': {'model': 'prepared-model', 'temperature': 0.7},
            'template': {'use_tokenizer_template': True},
            'env': {'HQ_ROOT': str(work / 'hq'), 'HQ_STARTUP_TIMEOUT': '5', 'SPEC': 'mtp'}}))
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        env = {key: value for key, value in os.environ.items() if not key.startswith(('LOCALAI_', 'P2P'))}
        # Keep checks from adding bytecode to an otherwise pristine extraction.
        env.update(PYTHONDONTWRITEBYTECODE='1', LOCALAI_GALLERIES='[]', LOCALAI_BACKEND_GALLERIES='[]',
                   LOCALAI_AUTOLOAD_BACKEND_GALLERIES='false', LOCALAI_AUTOLOAD_GALLERIES='false',
                   LOCALAI_P2P='false', P2P='false', TOKEN='')
        command = [str(binary), 'run', '--address', f'127.0.0.1:{port}',
                   '--models-path', str(models), '--backends-path', str(work / 'backends'),
                   '--backends-system-path', str(work / 'system-backends'), '--data-path', str(work / 'data'),
                   '--localai-config-dir', str(work / 'configuration'), '--disable-web-ui', '--disable-mcp', '--p2p=false']
        opener = build_opener(ProxyHandler({}))

        def request(path, payload=None):
            data = json.dumps(payload).encode() if payload is not None else None
            return opener.open(Request(f'http://127.0.0.1:{port}' + path, data=data,
                                       headers={'Content-Type': 'application/json'}), timeout=20)

        with log_path.open('x') as log:
            process = subprocess.Popen(command, cwd=work, env=env, stdout=log, stderr=log, start_new_session=True)
            try:
                deadline = time.monotonic() + 30
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f'controller exited; see {log_path}')
                    try:
                        with request('/readyz') as response:
                            if response.status == 200:
                                break
                    except (URLError, TimeoutError):
                        pass
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f'controller readiness failed; see {log_path}')
                    time.sleep(0.1)
                payload = {'model': 'hq-probe', 'messages': [{'role': 'user', 'content': 'Hi'}]}
                with request('/v1/chat/completions', payload) as response:
                    result = json.load(response)
                assert result['choices'][0]['message']['content'] == 'hello', result
                payload['stream'] = True
                with request('/v1/chat/completions', payload) as response:
                    wire = response.read().decode()
                assert 'hello' in wire and '[DONE]' in wire, wire
                payload.pop('stream')
                payload['tools'] = [{'type': 'function', 'function': {'name': 'lookup', 'parameters': {'type': 'object', 'properties': {}}}}]
                with request('/v1/chat/completions', payload) as response:
                    message = json.load(response)['choices'][0]['message']
                assert message['tool_calls'][0]['function']['name'] == 'lookup', message
                assert (message.get('reasoning_content') or message.get('reasoning')) == 'fixture reasoning', message
                payload['messages'] += [message, {'role': 'tool', 'content': 'fixture result', 'tool_call_id': message['tool_calls'][0]['id']}]
                with request('/v1/chat/completions', payload) as response:
                    continuation = json.load(response)
                assert continuation['choices'][0]['message']['content'] == 'tool acknowledged', continuation
                print('LocalAI directory discovery -> YAML env -> gRPC -> chat/SSE/tools/reasoning/continuation OK')
            finally:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('binary', help='existing checksum-pinned LocalAI executable')
    parser.add_argument('--package', help='existing extracted runtime; default tests source adapter')
    args = parser.parse_args()
    main(args.binary, args.package)
