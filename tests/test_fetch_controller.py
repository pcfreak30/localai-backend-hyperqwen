import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.fetch_controller import fetch


class FetchControllerTests(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.TemporaryDirectory()
        self.addCleanup(self.work.cleanup)
        self.destination = Path(self.work.name) / 'controller/local-ai'
        self.body = b'test executable placeholder'
        self.pin = {'version': 'v-test', 'asset': 'local-ai',
                    'sha256': hashlib.sha256(self.body).hexdigest()}

    def test_download_is_checked_and_made_executable(self):
        with patch('tests.fetch_controller.urlopen', return_value=io.BytesIO(self.body)) as request:
            self.assertEqual(fetch(self.destination, self.pin), self.destination)
        request.assert_called_once_with('https://github.com/mudler/LocalAI/releases/download/v-test/local-ai')
        self.assertEqual(self.destination.read_bytes(), self.body)
        self.assertEqual(self.destination.stat().st_mode & 0o777, 0o755)

    def test_verified_existing_executable_is_reused_without_download(self):
        self.destination.parent.mkdir()
        self.destination.write_bytes(self.body)
        self.destination.chmod(0o755)
        with patch('tests.fetch_controller.urlopen') as request:
            self.assertEqual(fetch(self.destination, self.pin), self.destination)
        request.assert_not_called()

    def test_wrong_existing_file_is_not_overwritten(self):
        self.destination.parent.mkdir()
        self.destination.write_bytes(b'wrong file')
        with patch('tests.fetch_controller.urlopen') as request:
            with self.assertRaisesRegex(ValueError, 'controller checksum mismatch'):
                fetch(self.destination, self.pin)
        request.assert_not_called()
        self.assertEqual(self.destination.read_bytes(), b'wrong file')

    def test_wrong_download_is_not_installed(self):
        with patch('tests.fetch_controller.urlopen', return_value=io.BytesIO(b'wrong download')):
            with self.assertRaisesRegex(ValueError, 'download checksum mismatch'):
                fetch(self.destination, self.pin)
        self.assertFalse(self.destination.exists())
        self.assertEqual(len(list(self.destination.parent.glob('*.partial'))), 1)
