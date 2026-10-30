"""Translate typed LocalAI startup settings to existing native launcher controls."""
import math


# field, native flag, aliases, value kind
SETTINGS = (
    ('Quantization', '--quantization', ('-q',), str),
    ('LoadFormat', '--load-format', (), str),
    ('TensorParallelSize', '--tensor-parallel-size', ('-tp',), int),
    ('SwapSpace', '--swap-space', (), int),
    ('EnforceEager', '--enforce-eager', ('--no-enforce-eager',), bool),
    ('TrustRemoteCode', '--trust-remote-code', ('--no-trust-remote-code',), bool),
    ('DType', '--dtype', (), str),
    ('DisableLogStatus', '--disable-log-stats', ('--no-disable-log-stats',), bool),
    ('LimitImagePerPrompt', None, (), int),
    ('LimitVideoPerPrompt', None, (), int),
    ('LimitAudioPerPrompt', None, (), int),
)


def _remove(words, flags, boolean=False):
    result = []
    index = 0
    while index < len(words):
        word = words[index]
        if word.split('=', 1)[0].replace('_', '-') in flags:
            if '=' not in word and not boolean:
                index += 1
        else:
            result.append(word)
        index += 1
    return result


def apply_load_settings(request, environment):
    env = dict(environment)
    # Native launchers expand EXTRA_ARGS unquoted: shell quote characters are
    # literal here, not a second round of shell syntax (notably JSON values).
    words = env.get('EXTRA_ARGS', '').split()
    original = list(words)
    additions = []
    for field, flag, aliases, kind in SETTINGS:
        value = getattr(request, field)
        if not value:
            continue  # proto3 scalars cannot distinguish omitted from false/zero.
        if kind is int and value < 0:
            raise ValueError(f'{field} must be positive')
        if flag is None:
            continue
        if kind is str and (any(c.isspace() for c in value) or value.startswith('-')):
            raise ValueError(f'{field} must be a single native argument value')
        words = _remove(words, {flag, *aliases}, boolean=kind is bool)
        additions.append(flag)
        if kind is not bool:
            additions.append(str(value))

    limits = {media: getattr(request, field) for media, field in (
        ('image', 'LimitImagePerPrompt'), ('video', 'LimitVideoPerPrompt'),
        ('audio', 'LimitAudioPerPrompt')) if getattr(request, field)}
    if limits:
        # Preserve env limits for modalities not overridden by LocalAI.
        import json
        retained = []
        for index, word in enumerate(words):
            if word.split('=', 1)[0].replace('_', '-') == '--limit-mm-per-prompt':
                raw = word.split('=', 1)[1] if '=' in word else words[index + 1]
                retained.append(json.loads(raw))
        merged = {}
        for item in retained:
            merged.update(item)
        merged.update(limits)
        words = _remove(words, {'--limit-mm-per-prompt'})
        additions += ['--limit-mm-per-prompt', json.dumps(merged, separators=(',', ':'))]

    length = request.MaxModelLen or request.ContextSize
    if request.MaxModelLen < 0 or request.ContextSize < 0:
        raise ValueError('MaxModelLen and ContextSize must be positive')
    if length:
        env['MAX_LEN'] = env['DFLASH_MAX_LEN'] = str(length)
        words = _remove(words, {'--max-model-len'})
        additions += ['--max-model-len', str(length)]

    utilization = request.GPUMemoryUtilization
    if utilization:
        if not math.isfinite(utilization) or not 0 < utilization <= 1:
            raise ValueError('GPUMemoryUtilization must be finite and in (0, 1]')
        env['GPU_UTIL'] = str(utilization)
        env['KV_MEM'] = ''
        words = _remove(words, {'--gpu-memory-utilization', '--kv-cache-memory', '--kv-cache-memory-bytes'})
        additions += ['--gpu-memory-utilization', str(utilization)]
    if words != original or additions:
        env['EXTRA_ARGS'] = ' '.join(words + additions)
    return env
