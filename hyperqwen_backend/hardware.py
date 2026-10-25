"""Opt-in deployment policies; never select GPUs outside the supplied visibility."""
import json
import os
import shlex
import subprocess
import sys


class HardwareConfigurationError(ValueError):
    pass


def visible_gpu_count(environment):
    # Keep Torch out of the adapter process and use the same native dependencies
    # and CUDA visibility as the child. Do not load weights or create an engine.
    probe = subprocess.run(
        [sys.executable, '-P', '-c', 'import json, torch; print(json.dumps(torch.cuda.device_count()))'],
        env=environment, capture_output=True, text=True, timeout=60, check=True,
    )
    count = json.loads(probe.stdout)
    if type(count) is not int or count < 0:
        raise ValueError('invalid visible GPU count')
    return count


def apply_hardware_policy(environment):
    env = dict(environment)
    sharing = env.pop('HQ_GPU_SHARING', 'off')
    cache = env.pop('HQ_CACHE_BUDGET', 'native')
    if sharing not in ('off', 'auto'):
        raise HardwareConfigurationError('HQ_GPU_SHARING must be off or auto')
    if cache not in ('native', 'auto'):
        raise HardwareConfigurationError('HQ_CACHE_BUDGET must be native or auto')
    words = shlex.split(env.get('EXTRA_ARGS', ''))
    topology = {'--tensor-parallel-size', '-tp', '--pipeline-parallel-size', '-pp',
                '--data-parallel-size', '-dp', '--decode-context-parallel-size', '-dcp',
                '--prefill-context-parallel-size', '-pcp', '--data-parallel-size-local',
                '--data-parallel-rank', '--nnodes', '--node-rank'}
    explicit = any(word.split('=', 1)[0].replace('_', '-') in topology for word in words)
    if sharing == 'auto':
        if explicit:
            print('[hardware-policy] GPU sharing: explicit parallel settings preserved', file=sys.stderr)
        else:
            # This is the final inherited environment, including deployment masks.
            probe_env = dict(os.environ)
            probe_env.update(env)
            try:
                count = visible_gpu_count(probe_env)
            except (OSError, subprocess.SubprocessError, ValueError) as exc:
                raise HardwareConfigurationError('automatic GPU detection failed; configure explicit parallel settings') from exc
            if count == 0:
                raise HardwareConfigurationError('automatic GPU sharing found no visible CUDA GPUs')
            extra = env.get('EXTRA_ARGS', '')
            env['EXTRA_ARGS'] = f'{extra} --tensor-parallel-size {count}'.strip()
            print(f'[hardware-policy] GPU sharing: {count} visible GPU(s), tensor parallel size {count}; '
                  'visibility is not an idle-device check', file=sys.stderr)
    if cache == 'auto':
        if 'KV_MEM' in env:
            print('[hardware-policy] cache: explicit KV_MEM preserved', file=sys.stderr)
        else:
            # Empty is intentional: native launchers use ${KV_MEM-default}, not :-.
            env['KV_MEM'] = ''
            cache_flags = {'--kv-cache-memory', '--kv-cache-memory-bytes'}
            if any(word.split('=', 1)[0].replace('_', '-') in cache_flags for word in words):
                print('[hardware-policy] cache: explicit command-line budget preserved', file=sys.stderr)
            else:
                print('[hardware-policy] cache: fixed budget disabled; native GPU_UTIL budgeting', file=sys.stderr)
    return env
