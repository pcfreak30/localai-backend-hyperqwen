import json
import os
from pathlib import Path
import tempfile
import unittest
import threading
from unittest.mock import patch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from hyperqwen_backend.runtime import ConfigurationError, Runtime, launch_configuration, positive_seconds
from tests.checkpoint_fixture import create_checkpoint


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.model = self.root / "model"
        self.model.mkdir()
        create_checkpoint(self.model)
        for script in ("single-user/start_qwen.sh", "batch/start_qwen.sh"):
            path = self.root / script
            path.parent.mkdir()
            path.write_text("exit 0\n")

    def test_explicit_model_and_native_launcher(self):
        command, env = launch_configuration(self.root, self.model, {"SPEC": "mtp", "PRINT_ARGV": "1", "HOST": "0.0.0.0"})
        self.assertEqual(command, ["bash", str(self.root / "single-user/start_qwen.sh")])
        self.assertEqual(env["MODEL"], str(self.model))
        self.assertEqual(env["HOST"], "127.0.0.1")
        self.assertNotIn("PRINT_ARGV", env)
        self.assertEqual(env["SPEC"], "mtp")
        self.assertNotIn("PTH_FA", env)

    def test_empty_environment_value_is_preserved(self):
        _, env = launch_configuration(self.root, self.model, {"INT8_ACT": ""})
        self.assertEqual(env["INT8_ACT"], "")

    def test_batch(self):
        command, env = launch_configuration(self.root, self.model, {"HQ_PROFILE": "batch"})
        self.assertTrue(command[-1].endswith("batch/start_qwen.sh"))
        self.assertNotIn("HQ_PROFILE", env)

    def test_reserved_extra_args(self):
        for value in ("--host=0.0.0.0", "--served_model_name other", "--port 9000"):
            with self.assertRaises(ConfigurationError):
                launch_configuration(self.root, self.model, {"EXTRA_ARGS": value})

    def test_shared_model_gpu_settings_reach_both_native_launchers(self):
        for profile in ('single', 'batch'):
            for count in (2, 4):
                with self.subTest(profile=profile, count=count):
                    devices = ','.join(str(index) for index in range(count))
                    extra = ('--enforce-eager ' if profile == 'batch' else '')
                    extra += '--tensor-parallel-size ' + str(count)
                    _, env = launch_configuration(self.root, self.model, {
                        'HQ_PROFILE': profile,
                        'CUDA_VISIBLE_DEVICES': devices,
                        'EXTRA_ARGS': extra,
                    })
                    self.assertEqual(env['CUDA_VISIBLE_DEVICES'], devices)
                    self.assertEqual(env['EXTRA_ARGS'], extra)

    def test_separate_model_copies_keep_their_gpu_selection(self):
        selected = []
        for device in ('0', '1'):
            _, env = launch_configuration(self.root, self.model, {
                'CUDA_VISIBLE_DEVICES': device,
            })
            selected.append(env)
        self.assertEqual(selected[0]['CUDA_VISIBLE_DEVICES'], '0')
        self.assertEqual(selected[1]['CUDA_VISIBLE_DEVICES'], '1')
        self.assertNotIn('EXTRA_ARGS', selected[0])
        self.assertNotIn('EXTRA_ARGS', selected[1])

    def test_native_dependency_path(self):
        _, env = launch_configuration(self.root, self.model, {"PYTHONPATH": "adapter", "HQ_NATIVE_PYTHONPATH": "native"})
        self.assertEqual(env["PYTHONPATH"], "native")

    def test_auto_detection_uses_native_dependencies_before_launch(self):
        for profile in ('single', 'batch'):
            with patch('hyperqwen_backend.hardware.visible_gpu_count', return_value=2) as probe:
                _, env = launch_configuration(self.root, self.model, {
                    'HQ_PROFILE': profile, 'HQ_GPU_SHARING': 'auto',
                    'HQ_NATIVE_PYTHONPATH': 'native', 'PYTHONPATH': 'adapter',
                    'CUDA_VISIBLE_DEVICES': 'selected-devices',
                })
                self.assertEqual(probe.call_args.args[0]['PYTHONPATH'], 'native')
                self.assertEqual(probe.call_args.args[0]['CUDA_VISIBLE_DEVICES'], 'selected-devices')
                self.assertEqual(env['EXTRA_ARGS'], '--tensor-parallel-size 2')
        with patch('hyperqwen_backend.hardware.visible_gpu_count', return_value=0):
            with self.assertRaises(ConfigurationError):
                launch_configuration(self.root, self.model, {'HQ_GPU_SHARING': 'auto'})

    def test_timeout_values(self):
        self.assertEqual(positive_seconds('2.5', 'timeout'), 2.5)
        for value in ('0', '-1', 'NaN', 'inf', 'oops'):
            with self.assertRaises(ConfigurationError):
                positive_seconds(value, 'timeout')

    def test_invalid_profile(self):
        with self.assertRaises(ConfigurationError):
            launch_configuration(self.root, self.model, {"HQ_PROFILE": "oops"})

    def test_wrong_architecture(self):
        (self.model / "config.json").write_text('{"architectures":["OtherModel"]}')
        with self.assertRaises(ConfigurationError):
            launch_configuration(self.root, self.model, {})

    def test_model_is_not_mutated(self):
        before = {p.name: p.read_bytes() for p in self.model.iterdir()}
        launch_configuration(self.root, self.model, {})
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.model.iterdir()})


class LifecycleTests(unittest.TestCase):
    def test_stop_and_restart(self):
        runtime = Runtime()
        self.addCleanup(runtime.stop)
        runtime.start(["sleep", "60"], dict(os.environ), "/tmp")
        process = runtime.process
        with self.assertRaises(RuntimeError):
            runtime.start(["sleep", "60"], dict(os.environ), "/tmp")
        runtime.stop()
        self.assertIsNotNone(process.returncode)
        runtime.stop()
        runtime.start(["sleep", "60"], dict(os.environ), "/tmp")

    def test_readiness_requires_authenticated_expected_model(self):
        state = {'model': 'wrong', 'authenticated': False}
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_GET(self):
                state['authenticated'] = self.headers.get('Authorization') == 'Bearer fixture-key'
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps({'data': [{'id': state['model']}]}).encode())
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        runtime = Runtime()
        self.addCleanup(runtime.stop)
        runtime.start(['sleep', '60'], dict(os.environ), '/tmp')
        url = f'http://127.0.0.1:{server.server_port}'
        with self.assertRaises(TimeoutError):
            runtime.wait_ready(url, 'fixture-key', timeout=0.05)
        self.assertTrue(state['authenticated'])
        state['model'] = 'qwen3.8-27b'
        runtime.wait_ready(url, 'fixture-key', timeout=1)

    def test_early_exit_is_not_ready(self):
        runtime = Runtime()
        self.addCleanup(runtime.stop)
        runtime.start(["false"], dict(os.environ), "/tmp")
        runtime.process.wait()
        with self.assertRaises(RuntimeError):
            runtime.wait_ready("http://127.0.0.1:1", "test", timeout=1)


if __name__ == "__main__":
    unittest.main()
