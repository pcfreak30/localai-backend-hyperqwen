import json
import unittest

import backend_pb2 as pb
from hyperqwen_backend.load_settings import apply_load_settings


class LoadSettingsTests(unittest.TestCase):
    def test_unset_preserves_environment(self):
        env = {'GPU_UTIL': '0.9', 'KV_MEM': '123', 'EXTRA_ARGS': '--dtype=half'}
        self.assertEqual(apply_load_settings(pb.ModelOptions(), env), env)

    def test_gpu_overrides_fixed_cache_and_arguments(self):
        env = {'GPU_UTIL': '0.9', 'KV_MEM': '123', 'EXTRA_ARGS':
               '--gpu_memory_utilization=.9 --kv-cache-memory 123 --kv-cache-memory-bytes=456 --foo bar'}
        result = apply_load_settings(pb.ModelOptions(GPUMemoryUtilization=.5), env)
        self.assertEqual(result['GPU_UTIL'], '0.5')
        self.assertEqual(result['KV_MEM'], '')
        self.assertEqual(result['EXTRA_ARGS'], '--foo bar --gpu-memory-utilization 0.5')
        self.assertEqual(env['KV_MEM'], '123')

    def test_context_precedence(self):
        for field in ('ContextSize', 'MaxModelLen'):
            result = apply_load_settings(pb.ModelOptions(**{field: 8192}),
                                         {'MAX_LEN': '100', 'DFLASH_MAX_LEN': '200',
                                          'EXTRA_ARGS': '--max-model-len=300'})
            self.assertEqual(result['MAX_LEN'], '8192')
            self.assertEqual(result['DFLASH_MAX_LEN'], '8192')
            self.assertEqual(result['EXTRA_ARGS'], '--max-model-len 8192')
        result = apply_load_settings(pb.ModelOptions(ContextSize=8192, MaxModelLen=4096), {})
        self.assertEqual(result['MAX_LEN'], '4096')

    def test_native_fields_override_aliases(self):
        request = pb.ModelOptions(Quantization='awq', LoadFormat='safetensors',
                                  TensorParallelSize=2, SwapSpace=4, EnforceEager=True,
                                  TrustRemoteCode=True, DType='half', DisableLogStatus=True)
        result = apply_load_settings(request, {'EXTRA_ARGS':
            '-tp=4 --dtype=bfloat16 --no-enforce-eager --no-trust-remote-code -q gptq'})
        args = result['EXTRA_ARGS']
        for flag in ('--quantization awq', '--load-format safetensors',
                     '--tensor-parallel-size 2', '--swap-space 4', '--enforce-eager',
                     '--trust-remote-code', '--dtype half', '--disable-log-stats'):
            self.assertIn(flag, args)
        self.assertNotIn('-tp=', args)
        self.assertNotIn('--no-', args)
        self.assertNotIn('gptq', args)

    def test_modality_limits_merge(self):
        result = apply_load_settings(pb.ModelOptions(LimitImagePerPrompt=2),
            {'EXTRA_ARGS': '--limit-mm-per-prompt={"image":4,"video":1}'})
        self.assertEqual(json.loads(result['EXTRA_ARGS'].split(' ', 1)[1]),
                         {'image': 2, 'video': 1})

    def test_invalid_values(self):
        for field, value in (('GPUMemoryUtilization', -1), ('GPUMemoryUtilization', 1.1),
                             ('GPUMemoryUtilization', float('nan')),
                             ('GPUMemoryUtilization', float('inf')),
                             ('TensorParallelSize', -1), ('ContextSize', -1),
                             ('SwapSpace', -1), ('DType', '--host')):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                apply_load_settings(pb.ModelOptions(**{field: value}), {})
