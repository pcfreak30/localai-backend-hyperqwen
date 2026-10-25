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

from tests.checkpoint_fixture import create_checkpoint, create_dflash_helper


def fixture_python(package):
    """Match the interpreter to the PYTHONHOME set by the package launcher."""
    if package:
        return str(Path(package).resolve() / 'python/bin/python3.12')
    return sys.executable


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
        helper = model / 'dflash'
        helper_source = models / 'fixture-helper'
        create_dflash_helper(helper_source)
        launcher = work / 'hq/single-user/start_qwen.sh'
        launcher.parent.mkdir(parents=True)
        launcher.write_text('set -eu\n'
                            + 'if [ "$SPEC" = dflash2 ]; then\n'
                            + '  test "$DRAFT" = ' + shlex.quote(str(helper)) + '\n'
                            + '  test "$DFLASH_TOKENS" = 7\n'
                            + 'fi\n'
                            + f'exec {shlex.quote(fixture_python(package))} {shlex.quote(str(root / "tests/fake_hq.py"))}\n')
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
        for name, spec in (('hq-probe', 'mtp'), ('hq-dflash-probe', 'dflash2')):
            startup = {'HQ_ROOT': str(work / 'hq'), 'HQ_STARTUP_TIMEOUT': '5', 'SPEC': spec}
            if spec == 'dflash2':
                startup.update(DRAFT='dflash', DFLASH_TOKENS='7')
            configuration = {
                'name': name, 'backend': 'hyperqwen',
                'parameters': {'model': 'prepared-model', 'temperature': 0.7},
                'template': {'use_tokenizer_template': True}, 'env': startup}
            if spec == 'mtp':
                (models / (name + '.yaml')).write_text(json.dumps(configuration))
                continue
            # LocalAI roots gallery filenames at models/, not at parameters.model.
            files = []
            for filename in ('config.json', 'model.safetensors'):
                source = helper_source / filename
                files.append({'filename': 'prepared-model/dflash/' + filename,
                              'uri': source.as_uri(),
                              'sha256': hashlib.sha256(source.read_bytes()).hexdigest()})
            catalog = models / 'fixture-gallery.json'
            catalog.write_text(json.dumps([{'name': name, 'config_file': configuration, 'files': files}]))
            installed = subprocess.run([
                str(binary), 'models', 'install', 'fixture@' + name,
                '--models-path', str(models), '--galleries',
                json.dumps([{'name': 'fixture', 'url': catalog.as_uri()}]),
                '--backend-galleries', '[]', '--autoload-backend-galleries=false',
                '--disable-predownload-scan',
            ], env={key: value for key, value in os.environ.items()
                    if not key.startswith(('LOCALAI_', 'P2P'))},
                capture_output=True, text=True, timeout=30)
            if installed.returncode:
                raise RuntimeError('fixture gallery install failed:\n' + installed.stdout + installed.stderr)
            for file in files:
                assert hashlib.sha256((models / file['filename']).read_bytes()).hexdigest() == file['sha256']
            print('LocalAI installed nested helper files with SHA256 checks', flush=True)
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        env = {key: value for key, value in os.environ.items() if not key.startswith(('LOCALAI_', 'P2P'))}
        # Bytecode files would change the extraction used for later checks.
        env.update(PYTHONDONTWRITEBYTECODE='1', LOCALAI_GALLERIES='[]', LOCALAI_BACKEND_GALLERIES='[]',
                   LOCALAI_AUTOLOAD_BACKEND_GALLERIES='false', LOCALAI_AUTOLOAD_GALLERIES='false', TOKEN='')
        command = [str(binary), 'run', '--address', f'127.0.0.1:{port}',
                   '--models-path', str(models), '--backends-path', str(work / 'backends'),
                   '--backends-system-path', str(work / 'system-backends'), '--data-path', str(work / 'data'),
                   '--localai-config-dir', str(work / 'configuration'), '--disable-web-ui', '--disable-mcp', '--p2p=false', '--p2p-token=']
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
                with request('/v1/chat/completions', {
                    'model': 'hq-dflash-probe', 'messages': [{'role': 'user', 'content': 'Hi'}]
                }) as response:
                    helper_result = json.load(response)
                assert helper_result['choices'][0]['message']['content'] == 'hello', helper_result
                print('LocalAI chat, streaming, tools, reasoning, tool replies and DFlash helper paths passed')
            except Exception:
                log.flush()
                print(log_path.read_text(), file=sys.stderr, flush=True)
                raise
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
