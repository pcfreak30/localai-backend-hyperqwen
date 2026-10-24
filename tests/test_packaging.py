"""Small source-level packaging guards; no Docker daemon or downloads."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PackagingTests(unittest.TestCase):
    def test_build_uses_pinned_native_image(self):
        lock = json.loads((ROOT / 'runtime.lock.json').read_text())
        self.assertIn('ARG HQ_IMAGE=' + lock['hyperqwen_image'], (ROOT / 'Dockerfile').read_text())
        self.assertEqual(lock['platform'], 'linux/amd64')
        ignore = (ROOT / '.dockerignore').read_text()
        self.assertIn('**/__pycache__/', ignore)
        self.assertIn('**/*.py[cod]', ignore)

    def test_protocol_matches_controller_pin(self):
        lock = json.loads((ROOT / 'runtime.lock.json').read_text())
        digest = hashlib.sha256((ROOT / 'protocol/backend.proto').read_bytes()).hexdigest()
        self.assertEqual(digest, lock['tested_controller']['protocol_sha256'])

    def test_extract_refuses_existing_directory_without_modifying_it(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / 'keep'
            marker.write_text('preserved')
            result = subprocess.run(['sh', str(ROOT / 'packaging/extract.sh'), 'unused-image', directory],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn('Refusing to overwrite', result.stderr)
            self.assertEqual(marker.read_text(), 'preserved')
            self.assertEqual(list(Path(directory).iterdir()), [marker])
