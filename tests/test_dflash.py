"""DFlash helper path and metadata checks; no real weights or GPU use."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hyperqwen_backend.runtime import ConfigurationError, launch_configuration
from tests.checkpoint_fixture import create_checkpoint, create_dflash_helper


class DFlashTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.model = self.root / 'model'
        self.model.mkdir()
        create_checkpoint(self.model)
        self.helper = self.model / 'dflash'
        create_dflash_helper(self.helper)
        for script in ('single-user/start_qwen.sh', 'batch/start_qwen.sh'):
            path = self.root / script
            path.parent.mkdir()
            path.write_text('exit 0\n')

    def launch(self, **overrides):
        env = {'SPEC': 'dflash2', 'DRAFT': 'dflash', **overrides}
        return launch_configuration(self.root, self.model, env)

    def test_relative_helper_and_controls_reach_native_launcher(self):
        original = {'SPEC': 'dflash2', 'DRAFT': 'dflash', 'DFLASH_TOKENS': '7',
                    'LOOKUP': '1', 'ASYNC_SCHED': '0',
                    'EXTRA_ARGS': '--tensor-parallel-size 2'}
        command, env = launch_configuration(self.root, self.model, original)
        self.assertEqual(command, ['bash', str(self.root / 'single-user/start_qwen.sh')])
        self.assertEqual(env['DRAFT'], str(self.helper))
        self.assertEqual(original['DRAFT'], 'dflash')
        for key in original.keys() - {'DRAFT'}:
            self.assertEqual(env[key], original[key])

    def test_absolute_helper_can_be_outside_target_model(self):
        external = self.root / 'separate-helper'
        create_dflash_helper(external)
        _, env = self.launch(DRAFT=str(external))
        self.assertEqual(env['DRAFT'], str(external))

    def test_relative_paths_cannot_escape_the_model_directory(self):
        external = self.root / 'separate-helper'
        create_dflash_helper(external)
        (self.model / 'escape').symlink_to(external, target_is_directory=True)
        for path in ('../separate-helper', 'escape'):
            with self.subTest(path=path), self.assertRaisesRegex(ConfigurationError, 'escapes'):
                self.launch(DRAFT=path)

    def test_cache_file_symlinks_are_supported_without_reading_tensors(self):
        payload = self.helper / 'model.safetensors'
        blob = self.root / 'cached-blob'
        payload.rename(blob)
        payload.symlink_to(blob)
        original_open = Path.open

        def metadata_only(path, *args, **kwargs):
            self.assertEqual(path.suffix, '.json', str(path))
            return original_open(path, *args, **kwargs)

        with patch.object(Path, 'open', metadata_only):
            self.launch()

    def test_missing_or_empty_helper_files_fail_before_launch(self):
        for name in ('config.json', 'model.safetensors'):
            path = self.helper / name
            data = path.read_bytes()
            for missing in (False, True):
                with self.subTest(name=name, missing=missing):
                    path.unlink() if missing else path.write_bytes(b'')
                    with self.assertRaises(ConfigurationError):
                        self.launch()
                    path.write_bytes(data)

    def test_invalid_helper_metadata_is_rejected(self):
        for config in ([], {'architectures': ['OtherModel']},
                       {'architectures': ['DFlash2DraftModel']},
                       {'architectures': ['DFlash2DraftModel'], 'dflash_config': {}},
                       {'architectures': ['DFlash2DraftModel'], 'dflash_config': []}):
            with self.subTest(config=config):
                (self.helper / 'config.json').write_text(json.dumps(config))
                with self.assertRaises(ConfigurationError):
                    self.launch()

    def test_invalid_paths_report_configuration_errors(self):
        (self.model / 'loop').symlink_to('loop')
        for path in ('missing', 'loop', 'bad\x00path'):
            with self.subTest(path=path), self.assertRaises(ConfigurationError):
                self.launch(DRAFT=path)

    def test_native_helper_discovery_is_unchanged_when_path_is_unset(self):
        for value in (None, ''):
            env = {'SPEC': 'dflash2'}
            if value is not None:
                env['DRAFT'] = value
            _, result = launch_configuration(self.root, self.model, env)
            self.assertEqual(result.get('DRAFT'), value)

    def test_other_native_modes_keep_their_existing_helper_paths(self):
        for overrides in ({'SPEC': 'mtp'}, {'HQ_PROFILE': 'batch'},
                          {'DRAFT_METHOD': 'dspark'}):
            with self.subTest(overrides=overrides):
                _, env = self.launch(DRAFT='missing', **overrides)
                self.assertEqual(env['DRAFT'], 'missing')

    def test_empty_draft_method_uses_the_native_default(self):
        _, env = self.launch(DRAFT_METHOD='')
        self.assertEqual(env['DRAFT'], str(self.helper))

    def test_dflash_does_not_require_the_targets_mtp_vocabulary(self):
        (self.model / 'mtp_draft_vocab_ids.pt').unlink()
        _, env = self.launch()
        self.assertEqual(env['SPEC'], 'dflash2')
