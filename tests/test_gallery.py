"""Check committed gallery metadata without fetching model payloads."""
import json
from pathlib import Path, PurePosixPath
import unittest

ROOT = Path(__file__).resolve().parents[1]
REVISION = '2e6ad7769c0273e95813090acd147ca9a70c686f'
MODEL = 'hyperqwen/Swift-1.5-Qwen3.8-27B-W4A16-HyperQwen'
UPSTREAM = 'https://huggingface.co/daavidhauser/Swift-1.5-Qwen3.8-27B-W4A16-HyperQwen/resolve/' + REVISION + '/'
HELPER_REVISION = '4d30ec736ffc6b8688dc2ae2b502d9b48bdec279'
HELPER_UPSTREAM = 'https://huggingface.co/syvai/Qwen3.8-27B-DFlash2-W4A16/resolve/' + HELPER_REVISION + '/'
HELPER_HASHES = {
    'config.json': '61d6276fe8d76295232cb02d26cbb0d29c25565911f50441e779c88c9220c556',
    'model.safetensors': 'ec26996e6a0745ab5edb857117220ce1e219ad524f71e6e149b703804947d8e7',
}
PROFILES = {}
for name, context, sequences in (
    ('swift-chat', 32768, 1), ('swift-long-conversations', 131072, 1),
    ('swift-multiple-conversations', 32768, 4),
    ('swift-multiple-long-conversations', 131072, 4),
):
    PROFILES[name] = (context, sequences, False)
    PROFILES[name + '-with-helper'] = (context, sequences, True)


class GalleryTests(unittest.TestCase):
    def setUp(self):
        self.models = json.loads((ROOT / 'gallery/models.yaml').read_text())

    def test_profiles_cover_length_conversations_and_helper_choices(self):
        self.assertEqual([item['name'] for item in self.models], list(PROFILES))
        for item in self.models:
            context, sequences, helper = PROFILES[item['name']]
            config = item['config_file']
            env = config['env']
            self.assertEqual(config['name'], item['name'])
            self.assertEqual(config['backend'], 'hyperqwen')
            self.assertEqual(config['parameters']['model'], MODEL)
            self.assertTrue(config['template']['use_tokenizer_template'])
            self.assertEqual(config['context_size'], context)
            self.assertEqual(env['MAX_LEN'], str(context))
            self.assertEqual(env['MAX_SEQS'], str(sequences))
            # The old batch mode ignores SPEC. Both choices must use the single launcher.
            self.assertEqual(env['HQ_PROFILE'], 'single')
            self.assertEqual(env['CTX'], 'long')
            self.assertEqual(env['HQ_GPU_SHARING'], 'auto')
            self.assertEqual(env['HQ_CACHE_BUDGET'], 'native')
            self.assertIn('Shares all exposed GPUs, including busy ones', item['description'])
            self.assertIn('Upgrade the HyperQwen backend', item['description'])
            self.assertTrue(all(isinstance(value, str) for value in env.values()))
            if sequences == 4:
                self.assertEqual(env['EXTRA_ARGS'], '--enforce-eager')
            if helper:
                self.assertEqual(env['SPEC'], 'dflash2')
                self.assertEqual(env['DRAFT_METHOD'], 'dflash')
                self.assertEqual(env['DRAFT'], 'dflash')
                self.assertEqual(env['DFLASH_TOKENS'], '7')
                self.assertEqual(env['LOOKUP'], '1')
                self.assertEqual(env['ASYNC_SCHED'], '0')
                self.assertNotIn('DRAFT_TOKENS', env)
            else:
                self.assertEqual(env['SPEC'], 'mtp')
                self.assertEqual(env['DRAFT_TOKENS'], '2')
                self.assertNotIn('DRAFT', env)

    def test_profiles_leave_hardware_selection_to_deployment(self):
        for item in self.models:
            env = item['config_file']['env']
            for key in ('CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES',
                        'PTH_FA', 'FA2_FP8KV', 'GPU_UTIL', 'GPU_COUNT', 'HQ_ROOT'):
                self.assertNotIn(key, env)
            self.assertNotIn('tensor-parallel-size', env.get('EXTRA_ARGS', ''))
            target = item['config_file']['parameters']['model'].lower()
            for inferred_method in ('dflash', 'dspark', 'eagle3', 'eagle-'):
                self.assertNotIn(inferred_method, target)
            text = ' '.join([item['name'], item['description'], *item['tags']]).lower()
            for term in ('3090', 'batch', 'graphs', 'mtp', 'dflash', 'gpu-'):
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

    def test_prepared_target_artifacts_remain_shared_and_pinned(self):
        base = self.models[0]['files']
        self.assertEqual(len(base), 25)
        for item in self.models:
            helper = PROFILES[item['name']][2]
            self.assertEqual(item['files'][:25], base)
            self.assertEqual(len(item['files']), 27 if helper else 25)
        self.assertEqual(sum(f['filename'].endswith('.safetensors') for f in base), 8)
        for required in ('config.json', 'model.safetensors.index.json', 'tokenizer.json',
                         'tokenizer_config.json', 'mtp_draft_vocab_ids.pt'):
            self.assertIn(MODEL + '/' + required, [f['filename'] for f in base])
        for item in base:
            self.assertEqual(item['uri'], UPSTREAM + item['filename'][len(MODEL) + 1:])

    def test_helper_download_targets_match_the_model_relative_path(self):
        for item in self.models:
            if not PROFILES[item['name']][2]:
                continue
            config = item['config_file']
            # Gallery filenames are rooted at LocalAI's models directory, not at parameters.model.
            destination = PurePosixPath(config['parameters']['model']) / config['env']['DRAFT']
            self.assertEqual(len(item['files'][25:]), 2)
            for file in item['files'][25:]:
                name = PurePosixPath(file['filename']).name
                self.assertIn(name, HELPER_HASHES)
                self.assertEqual(file['filename'], str(destination / name))
                self.assertEqual(file['sha256'], HELPER_HASHES[name])
                self.assertEqual(file['uri'], HELPER_UPSTREAM + name)
            # Downloads belong to gallery installation, not a second config-preload mechanism.
            self.assertNotIn('download_files', config)
            self.assertEqual(item['size'], '18143935192B')

    def test_all_artifacts_have_safe_names_hashes_and_no_executable_payloads(self):
        for model in self.models:
            names = [item['filename'] for item in model['files']]
            self.assertEqual(len(names), len(set(names)))
            for item in model['files']:
                path = PurePosixPath(item['filename'])
                self.assertFalse(path.is_absolute())
                self.assertNotIn('..', path.parts)
                self.assertTrue(item['filename'].startswith(MODEL + '/'))
                self.assertNotIn(path.suffix, ('.py', '.sh', '.patch', '.so'))
                self.assertRegex(item['sha256'], r'^[0-9a-f]{64}$')

    def test_backend_entry_uses_the_reviewed_update_channel(self):
        entries = json.loads((ROOT / 'gallery/backends.yaml').read_text())
        self.assertEqual(len(entries), 1)
        for entry in entries:
            self.assertEqual(entry['name'], 'hyperqwen')
            self.assertIsInstance(entry['description'], str)
            self.assertTrue(entry['description'])
            self.assertEqual(entry['uri'], 'oci://ghcr.io/pcfreak30/localai-backend-hyperqwen:latest')
        release = (ROOT / '.github/workflows/release.yml').read_text()
        self.assertLess(release.index('test "$GPU_REVIEWED" = true'),
                        release.index('name: Promote the verified digest to latest'))
        self.assertIn('test "$promoted" = "$DIGEST"', release)
