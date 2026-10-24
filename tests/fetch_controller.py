"""Fetch the pinned LocalAI test executable and verify its SHA256 before use."""
import argparse
import hashlib
import json
from pathlib import Path
import os
import shutil
import tempfile
from urllib.request import urlopen


def fetch(destination, pin):
    destination = Path(destination)
    if destination.exists():
        with destination.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != pin['sha256']:
            raise ValueError(f'controller checksum mismatch: {destination}')
        if not os.access(destination, os.X_OK):
            raise ValueError(f'controller is not executable: {destination}')
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f'https://github.com/mudler/LocalAI/releases/download/{pin["version"]}/{pin["asset"]}'
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=destination.name + '.', suffix='.partial', delete=False) as output:
        partial = Path(output.name)
        with urlopen(url) as response:
            shutil.copyfileobj(response, output)
    with partial.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != pin['sha256']:
        raise ValueError(f'download checksum mismatch; retained {partial}')
    partial.chmod(0o755)
    os.replace(partial, destination)
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    lock = Path(__file__).resolve().parents[1] / 'runtime.lock.json'
    print(fetch(args.destination, json.loads(lock.read_text())['tested_controller']))
