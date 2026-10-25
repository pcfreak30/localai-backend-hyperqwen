"""Read-only metadata/file preflight for prepared HQ checkpoints and helpers.

Never opens tensor files, deserializes pickle, executes model code or downloads.
Native HQ still owns hardware/plugin validation and tensor loading.
"""
from __future__ import annotations

import json
from pathlib import Path, PurePosixPath


class CheckpointError(ValueError):
    pass


def _object(path: Path):
    try:
        with path.open('rb') as stream:
            data = stream.read(32 * 1024 * 1024 + 1)
        if len(data) > 32 * 1024 * 1024:
            raise ValueError('metadata exceeds 32 MiB')
        value = json.loads(data)
        if not isinstance(value, dict):
            raise ValueError('expected a JSON object')
        return value
    except (OSError, ValueError) as exc:
        raise CheckpointError(f'{path.name} requires a readable JSON object: {exc}') from exc


def _file(path: Path):
    try:
        valid = path.is_file() and path.stat().st_size > 0
    except OSError:
        valid = False
    if not valid:
        raise CheckpointError(f'missing or empty checkpoint file: {path.name}')


def validate_dflash_helper(helper: Path):
    """Check the single-file DFlash2 helper without opening its tensor payload."""
    config = _object(helper / 'config.json')
    if config.get('architectures') != ['DFlash2DraftModel']:
        raise CheckpointError('DFlash helper requires the DFlash2DraftModel architecture')
    if not isinstance(config.get('dflash_config'), dict) or not config['dflash_config']:
        raise CheckpointError('DFlash helper requires dflash_config metadata')
    _file(helper / 'model.safetensors')


def validate_checkpoint(model: Path, environment: dict[str, str]):
    """Validate the initial indexed compressed-tensors contract using metadata only.

    HF cache symlinks are supported. Index filenames may not lexically escape the
    model directory, but cache symlinks may legitimately point to shared blobs.
    """
    config = _object(model / 'config.json')
    architectures = config.get('architectures')
    supported = {'Qwen3_5ForConditionalGeneration', 'Qwen3_8ForConditionalGeneration'}
    if not isinstance(architectures, list) or not architectures or not all(isinstance(x, str) for x in architectures) or not supported.intersection(architectures):
        raise CheckpointError(f'unsupported model architecture: {architectures}')
    quant = config.get('quantization_config')
    if not isinstance(quant, dict) or quant.get('quant_method') != 'compressed-tensors' or quant.get('format') != 'pack-quantized':
        raise CheckpointError('prepared checkpoint requires compressed-tensors / pack-quantized metadata')
    groups = quant.get('config_groups')
    if not isinstance(groups, dict) or not groups:
        raise CheckpointError('quantization config_groups must be a nonempty object')
    for name, group in groups.items():
        if not isinstance(group, dict):
            raise CheckpointError(f'invalid quantization group: {name}')
        weights, targets = group.get('weights'), group.get('targets')
        if (not isinstance(weights, dict) or weights.get('type') != 'int'
                or weights.get('num_bits') not in (4, 8)
                or not isinstance(weights.get('symmetric'), bool)
                or not isinstance(targets, list) or not targets
                or not all(isinstance(target, str) and target for target in targets)):
            raise CheckpointError(f'quantization group {name} requires int4/int8 weights and named targets')
    tokenizer = _object(model / 'tokenizer_config.json')
    _file(model / 'tokenizer.json')
    if not tokenizer.get('chat_template'):
        _file(model / 'chat_template.jinja')
    index = _object(model / 'model.safetensors.index.json')
    mapping = index.get('weight_map')
    if not isinstance(mapping, dict) or not mapping:
        raise CheckpointError('safetensors index requires a nonempty weight_map')
    for tensor, filename in mapping.items():
        if not isinstance(tensor, str) or not tensor or not isinstance(filename, str):
            raise CheckpointError('weight_map requires named tensors and shard filenames')
        path = PurePosixPath(filename)
        if (not filename or path.is_absolute() or '..' in path.parts or '\\' in filename
                or '\x00' in filename or path.suffix != '.safetensors'):
            raise CheckpointError(f'unsafe safetensors shard filename: {filename!r}')
    for filename in sorted(set(mapping.values())):
        _file(model / filename)
    packed = [name.removesuffix('.weight_packed') for name in mapping if name.endswith('.weight_packed')]
    if not packed:
        raise CheckpointError('pack-quantized index has no weight_packed tensors')
    for name in packed:
        for suffix in ('.weight_scale', '.weight_shape'):
            if name + suffix not in mapping:
                raise CheckpointError(f'packed tensor is missing indexed companion: {name + suffix}')
    # Match the pinned native single launcher default, without changing its env.
    # Batch does not use the single-user speculative path.
    spec = (environment.get('SPEC') or 'mtp') if environment.get('HQ_PROFILE', 'single') == 'single' else 'off'
    if spec == 'mtp':
        if not any(name.startswith('mtp.layers.') for name in mapping) or not any(name.startswith('mtp.fc.') for name in mapping):
            raise CheckpointError('SPEC=mtp requires indexed MTP layer and projection tensors')
        if any(name.startswith('mtp.draft_lm_head.') for name in mapping):
            _file(model / 'mtp_draft_vocab_ids.pt')
    return {'shards': len(set(mapping.values())), 'indexed_tensors': len(mapping), 'packed_modules': len(packed)}
