"""Check committed gallery metadata without fetching model payloads."""
import json
from pathlib import Path, PurePosixPath
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
REVISION = '2e6ad7769c0273e95813090acd147ca9a70c686f'
MODEL = 'hyperqwen/Swift-1.5-Qwen3.8-27B-W4A16-HyperQwen'
UPSTREAM = 'https://huggingface.co/daavidhauser/Swift-1.5-Qwen3.8-27B-W4A16-HyperQwen/resolve/' + REVISION + '/'


class GalleryTests(unittest.TestCase):
    def setUp(self):
        self.models = json.loads((ROOT / 'gallery/models.yaml').read_text())

    def test_profiles_use_4096_tokens_and_two_sequences(self):
        self.assertEqual([item['name'] for item in self.models],
                         ['swift-hyperqwen-3090-graphs', 'swift-hyperqwen-3090-batch'])
        for item in self.models:
            config = item['config_file']
            self.assertEqual(config['name'], item['name'])
            self.assertEqual(config['backend'], 'hyperqwen')
            self.assertEqual(config['parameters']['model'], MODEL)
            self.assertTrue(config['template']['use_tokenizer_template'])
            self.assertEqual(config['env']['MAX_LEN'], '4096')
            self.assertEqual(config['env']['MAX_SEQS'], '2')
            self.assertTrue(all(isinstance(value, str) for value in config['env'].values()))
        self.assertEqual(self.models[0]['config_file']['env']['SPEC'], 'mtp')
        self.assertEqual(self.models[1]['config_file']['env']['HQ_PROFILE'], 'batch')
        self.assertEqual(self.models[1]['config_file']['env']['INT8_ACT'], '')
        self.assertEqual(self.models[1]['config_file']['env']['EXTRA_ARGS'], '--enforce-eager')

    def test_prepared_artifacts_are_pinned_and_nonexecutable(self):
        files = self.models[0]['files']
        self.assertEqual(files, self.models[1]['files'])
        self.assertEqual(len(files), 25)
        names = [item['filename'] for item in files]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(sum(name.endswith('.safetensors') for name in names), 8)
        for required in ('config.json', 'model.safetensors.index.json', 'tokenizer.json',
                         'tokenizer_config.json', 'mtp_draft_vocab_ids.pt'):
            self.assertIn(MODEL + '/' + required, names)
        for item in files:
            path = PurePosixPath(item['filename'])
            self.assertFalse(path.is_absolute())
            self.assertNotIn('..', path.parts)
            self.assertTrue(item['filename'].startswith(MODEL + '/'))
            self.assertNotIn(path.suffix, ('.py', '.sh', '.patch', '.so'))
            self.assertRegex(item['sha256'], r'^[0-9a-f]{64}$')
            self.assertEqual(item['uri'], UPSTREAM + item['filename'][len(MODEL) + 1:])

    def test_backend_entry_uses_an_image_digest(self):
        entries = json.loads((ROOT / 'gallery/backends.yaml').read_text())
        self.assertEqual(len(entries), 1)
        for entry in entries:
            self.assertEqual(entry['name'], 'hyperqwen')
            self.assertIsInstance(entry['description'], str)
            self.assertTrue(entry['description'])
            self.assertIsNotNone(re.fullmatch(
                r'oci://ghcr\.io/pcfreak30/localai-backend-hyperqwen@sha256:[0-9a-f]{64}', entry['uri']))
