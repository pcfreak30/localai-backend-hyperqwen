import io
import json
from contextlib import redirect_stderr
import subprocess
import unittest
from unittest.mock import patch, Mock

from hyperqwen_backend.hardware import (
    HardwareConfigurationError, apply_hardware_policy, visible_gpu_count,
)


class HardwareTests(unittest.TestCase):
    def test_default_does_not_probe_or_change_native_settings(self):
        env = {'EXTRA_ARGS': '--enforce-eager', 'GPU_UTIL': '0.8'}
        with patch('hyperqwen_backend.hardware.visible_gpu_count') as probe:
            self.assertEqual(apply_hardware_policy(env), env)
            probe.assert_not_called()

    def test_auto_uses_visible_count_and_keeps_arguments(self):
        for count in (1, 2, 4):
            with self.subTest(count=count), patch('hyperqwen_backend.hardware.visible_gpu_count', return_value=count):
                original = {'HQ_GPU_SHARING': 'auto', 'EXTRA_ARGS': '--enforce-eager'}
                result = apply_hardware_policy(original)
                self.assertEqual(result['EXTRA_ARGS'], f'--enforce-eager --tensor-parallel-size {count}')
                self.assertNotIn('HQ_GPU_SHARING', result)
                self.assertEqual(original['EXTRA_ARGS'], '--enforce-eager')

    def test_explicit_topology_wins_without_probe(self):
        for flag in ('--tensor-parallel-size 2', '--tensor_parallel_size=2', '-tp=2',
                     '--pipeline-parallel-size 2', '--data-parallel-size=2',
                     '--decode-context-parallel-size 2', '--prefill-context-parallel-size 2'):
            with self.subTest(flag=flag), patch('hyperqwen_backend.hardware.visible_gpu_count') as probe:
                result = apply_hardware_policy({'HQ_GPU_SHARING': 'auto', 'EXTRA_ARGS': flag})
                self.assertEqual(result['EXTRA_ARGS'], flag)
                probe.assert_not_called()

    def test_masks_are_forwarded_unchanged(self):
        env = {'HQ_GPU_SHARING': 'auto', 'CUDA_VISIBLE_DEVICES': 'GPU-test',
               'NVIDIA_VISIBLE_DEVICES': 'GPU-test', 'PYTHONPATH': 'native-only'}
        with patch('hyperqwen_backend.hardware.visible_gpu_count', return_value=1) as probe:
            apply_hardware_policy(env)
            received = probe.call_args.args[0]
            for key in ('CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES', 'PYTHONPATH'):
                self.assertEqual(received[key], env[key])

    def test_no_devices_or_probe_failure_is_not_silently_ignored(self):
        with patch('hyperqwen_backend.hardware.visible_gpu_count', return_value=0):
            with self.assertRaisesRegex(HardwareConfigurationError, 'no visible'):
                apply_hardware_policy({'HQ_GPU_SHARING': 'auto'})
        for error in (OSError(), ValueError(), subprocess.TimeoutExpired('probe', 60),
                      subprocess.CalledProcessError(1, 'probe')):
            with self.subTest(error=type(error).__name__), patch('hyperqwen_backend.hardware.visible_gpu_count', side_effect=error):
                with self.assertRaisesRegex(HardwareConfigurationError, 'detection failed'):
                    apply_hardware_policy({'HQ_GPU_SHARING': 'auto'})

    def test_invalid_policies(self):
        for env in ({'HQ_GPU_SHARING': 'yes'}, {'HQ_CACHE_BUDGET': 'yes'}):
            with self.assertRaises(HardwareConfigurationError):
                apply_hardware_policy(env)

    def test_cache_auto_only_clears_an_unset_pin(self):
        result = apply_hardware_policy({'HQ_CACHE_BUDGET': 'auto', 'GPU_UTIL': '0.8'})
        self.assertEqual(result['KV_MEM'], '')
        self.assertEqual(result['GPU_UTIL'], '0.8')
        self.assertNotIn('HQ_CACHE_BUDGET', result)
        for value in ('12345', ''):
            result = apply_hardware_policy({'HQ_CACHE_BUDGET': 'auto', 'KV_MEM': value})
            self.assertEqual(result['KV_MEM'], value)

    def test_explicit_command_line_cache_budget_is_preserved_and_logged(self):
        log = io.StringIO()
        extra = '--kv-cache-memory=12345'
        with redirect_stderr(log):
            result = apply_hardware_policy({'HQ_CACHE_BUDGET': 'auto', 'EXTRA_ARGS': extra})
        self.assertEqual(result['EXTRA_ARGS'], extra)
        self.assertEqual(result['KV_MEM'], '')
        self.assertIn('explicit command-line budget preserved', log.getvalue())
        self.assertNotIn('GPU_UTIL budgeting', log.getvalue())

    def test_probe_uses_isolated_python_and_bounded_timeout(self):
        env = {'CUDA_VISIBLE_DEVICES': ''}
        with patch('hyperqwen_backend.hardware.subprocess.run', return_value=Mock(stdout='0\n')) as run:
            self.assertEqual(visible_gpu_count(env), 0)
            self.assertEqual(run.call_args.kwargs['env'], env)
            self.assertEqual(run.call_args.kwargs['timeout'], 60)
            self.assertTrue(run.call_args.kwargs['check'])
            self.assertIn('-P', run.call_args.args[0])
            self.assertIn('torch.cuda.device_count()', run.call_args.args[0][-1])

    def test_probe_rejects_invalid_output(self):
        for value in (-1, True, 1.5, '2', None):
            with self.subTest(value=value), patch('hyperqwen_backend.hardware.subprocess.run', return_value=Mock(stdout=json.dumps(value))):
                with self.assertRaises(ValueError):
                    visible_gpu_count({})
