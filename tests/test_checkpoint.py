import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hyperqwen_backend.checkpoint import CheckpointError, validate_checkpoint
from tests.checkpoint_fixture import create_checkpoint


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.model = Path(self.tmp.name)
        create_checkpoint(self.model)

    def change(self, name, mutate):
        path = self.model / name
        value = json.loads(path.read_text())
        mutate(value)
        path.write_text(json.dumps(value))

    def test_prepared_checkpoint_and_no_tensor_deserialization(self):
        original = Path.open
        def metadata_only(path, *args, **kwargs):
            if path.suffix in ('.safetensors', '.pt'):
                raise AssertionError('tensor contents must never be opened')
            return original(path, *args, **kwargs)
        with patch.object(Path, 'open', metadata_only):
            result = validate_checkpoint(self.model, {})
        self.assertEqual(result, {'shards': 1, 'indexed_tensors': 12, 'packed_modules': 4})

    def test_missing_files(self):
        for name in ('tokenizer.json', 'tokenizer_config.json', 'fixture.safetensors',
                     'model.safetensors.index.json', 'mtp_draft_vocab_ids.pt'):
            with self.subTest(name=name):
                create_checkpoint(self.model)
                # Preserve each fixture for the next subtest without deleting it.
                (self.model / name).rename(self.model / (name + '.saved'))
                with self.assertRaises(CheckpointError):
                    validate_checkpoint(self.model, {})
                (self.model / (name + '.saved')).rename(self.model / name)

    def test_malformed_config_and_index_objects(self):
        for name in ('config.json', 'tokenizer_config.json', 'model.safetensors.index.json'):
            for content in ('[]', 'null', '{broken'):
                with self.subTest(name=name, content=content):
                    create_checkpoint(self.model)
                    (self.model / name).write_text(content)
                    with self.assertRaises(CheckpointError):
                        validate_checkpoint(self.model, {})

    def test_strict_architecture_and_quantization(self):
        for mutate in (lambda c: c.update(architectures=['OtherQwen3_5Name']),
                       lambda c: c.update(architectures='Qwen3_5ForConditionalGeneration'),
                       lambda c: c.update(quantization_config={'quant_method': 'awq'}),
                       lambda c: c['quantization_config'].update(config_groups={}),
                       lambda c: c['quantization_config']['config_groups']['linear']['weights'].update(num_bits=3)):
            create_checkpoint(self.model)
            self.change('config.json', mutate)
            with self.assertRaises(CheckpointError):
                validate_checkpoint(self.model, {})

    def test_unsafe_index_paths_and_empty_maps(self):
        for filename in ('../outside.safetensors', '/absolute.safetensors', 'C:\\bad.safetensors',
                         'bad\x00.safetensors', 'script.py', ''):
            create_checkpoint(self.model)
            self.change('model.safetensors.index.json', lambda c: c.update(weight_map={'tensor': filename}))
            with self.assertRaisesRegex(CheckpointError, 'unsafe'):
                validate_checkpoint(self.model, {})
        self.change('model.safetensors.index.json', lambda c: c.update(weight_map={}))
        with self.assertRaisesRegex(CheckpointError, 'nonempty'):
            validate_checkpoint(self.model, {})

    def test_packed_tensor_companions(self):
        self.change('model.safetensors.index.json', lambda c: c['weight_map'].pop('model.layer.weight_scale'))
        with self.assertRaisesRegex(CheckpointError, 'companion'):
            validate_checkpoint(self.model, {})

    def test_mtp_is_required_only_for_selected_speculation(self):
        self.change('model.safetensors.index.json', lambda c: c.update(weight_map={
            key: value for key, value in c['weight_map'].items() if not key.startswith('mtp.')}))
        with self.assertRaisesRegex(CheckpointError, 'MTP'):
            validate_checkpoint(self.model, {})
        validate_checkpoint(self.model, {'SPEC': 'off'})
        validate_checkpoint(self.model, {'HQ_PROFILE': 'batch', 'SPEC': 'mtp'})

    def test_shared_cache_symlinks(self):
        shard = self.model / 'fixture.safetensors'
        target = self.model / 'shared-blob'
        shard.rename(target)
        shard.symlink_to(target)
        validate_checkpoint(self.model, {})

    def test_separate_chat_template(self):
        self.change('tokenizer_config.json', lambda c: c.pop('chat_template'))
        with self.assertRaisesRegex(CheckpointError, 'chat_template'):
            validate_checkpoint(self.model, {})
        (self.model / 'chat_template.jinja').write_text('{{ messages }}')
        validate_checkpoint(self.model, {})
