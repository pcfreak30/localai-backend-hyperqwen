from pathlib import Path
import unittest
from unittest.mock import patch

from tests.controller_smoke import fixture_python


class ControllerSmokeTests(unittest.TestCase):
    def test_package_fixture_uses_bundled_python_instead_of_host_python(self):
        package = Path('build/test-package').resolve()
        with patch('tests.controller_smoke.sys.executable', '/opt/hostedtoolcache/Python/bin/python'):
            self.assertEqual(fixture_python(package), str(package / 'python/bin/python3.12'))

    def test_source_fixture_uses_host_python(self):
        with patch('tests.controller_smoke.sys.executable', '/source/.venv/bin/python'):
            self.assertEqual(fixture_python(None), '/source/.venv/bin/python')
