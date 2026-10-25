"""Check committed gallery metadata without fetching model payloads."""
import json
from pathlib import Path, PurePosixPath
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
REVISION = '2e6ad7769c0273e95813090acd147ca9a70c686f'
MODEL = 'hyperqwen/Swift-1.5-Qwen3.8-27B-W4A16-HyperQwen'
UPSTREAM = 'https://huggingface.co/daavidhauser/Swift-1.5-Qwen3.8-27B-W4A16-HyperQwen/resolve/' + REVISION + '/'
PROFILES = {
    'swift-chat': (32768, 1, 'single'),
    'swift-long-conversations': (131072, 1, 'single'),
    'swift-multiple-conversations': (32768, 4, 'batch'),
    'swift-multiple-long-conversations': (131072, 4, 'batch'),
}


class GalleryTests(unittest.TestCase):
    def setUp(self):
        self.models = json.loads((ROOT / 'gallery/models.yaml').read_text())

    def test_profiles_cover_length_and_conversation_choices(self):
        self.assertEqual([item['name'] for item in self.models], list(PROFILES))
        for item in self.models:
            context, sequences, profile = PROFILES[item['name']]
            config = item['config_file']
            env = config['env']
            self.assertEqual(config['name'], item['name'])
            self.assertEqual(config['backend'], 'hyperqwen')
            self.assertEqual(config['parameters']['model'], MODEL)
            self.assertTrue(config['template']['use_tokenizer_template'])
            self.assertEqual(config['context_size'], context)
            self.assertEqual(env['MAX_LEN'], str(context))
            self.assertEqual(env['MAX_SEQS'], str(sequences))
            self.assertEqual(env['HQ_PROFILE'], profile)
            self.assertTrue(all(isinstance(value, str) for value in env.values()))
            if profile == 'single':
                self.assertEqual(env['SPEC'], 'mtp')
                self.assertEqual(env['CTX'], 'long')
            else:
                self.assertEqual(env['INT8_ACT'], '')
                self.assertEqual(env['EXTRA_ARGS'], '--enforce-eager')
                self.assertNotIn('SPEC', env)
                self.assertNotIn('CTX', env)

    def test_profiles_leave_hardware_selection_to_deployment(self):
        for item in self.models:
            env = item['config_file']['env']
            for key in ('CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES',
                        'PTH_FA', 'FA2_FP8KV', 'GPU_UTIL', 'HQ_ROOT'):
                self.assertNotIn(key, env)
            self.assertNotIn('tensor-parallel-size', env.get('EXTRA_ARGS', ''))
            text = ' '.join([item['name'], item['description'], *item['tags']]).lower()
            for term in ('3090', 'batch', 'graphs', 'mtp', 'gpu-'):
                self.assertNotIn(term, text)

    def test_examples_match_gallery_configuration(self):
        self.assertEqual({path.stem for path in (ROOT / 'examples').glob('*.yaml')},
                         set(PROFILES))
        for item in self.models:
            config = item['config_file']
            example = (ROOT / 'examples' / (item['name'] + '.yaml')).read_text()
            self.assertIn('name: ' + item['name'] + '\n', example)
            self.assertIn('context_size: ' + str(config['context_size']) + '\n', example)
            self.assertIn('  model: ' + MODEL.removeprefix('hyperqwen/') + '\n', example)
            for key, value in config['env'].items():
                self.assertIn('  ' + key + ': ' + json.dumps(value) + '\n', example)

    def test_prepared_artifacts_are_pinned_and_nonexecutable(self):
        files = self.models[0]['files']
        for model in self.models[1:]:
            self.assertEqual(files, model['files'])
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
