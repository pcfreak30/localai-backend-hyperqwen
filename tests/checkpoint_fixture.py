"""Synthetic checkpoint metadata for CPU tests; never actual model weights."""
import json


def create_checkpoint(model):
    config = {'architectures': ['Qwen3_5ForConditionalGeneration'],
              'quantization_config': {'quant_method': 'compressed-tensors', 'format': 'pack-quantized',
                                     'config_groups': {'linear': {'targets': ['Linear'], 'weights': {
                                         'type': 'int', 'num_bits': 4, 'symmetric': False}}}}}
    (model / 'config.json').write_text(json.dumps(config))
    (model / 'tokenizer_config.json').write_text(json.dumps({'chat_template': '{{ messages }}'}))
    (model / 'tokenizer.json').write_text('{}')
    mapping = {f'{module}.{suffix}': 'fixture.safetensors'
               for module in ('model.layer', 'mtp.layers.0.mlp', 'mtp.fc', 'mtp.draft_lm_head')
               for suffix in ('weight_packed', 'weight_scale', 'weight_shape')}
    (model / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': mapping}))
    for name in ('fixture.safetensors', 'mtp_draft_vocab_ids.pt'):
        (model / name).write_text('CPU TEST PLACEHOLDER: NOT MODEL WEIGHTS')
