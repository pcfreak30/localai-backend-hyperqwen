"""LocalAI 4.12.1 reads Sigstore bundles through OCI referrers, not .sig tags."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class SigningWorkflowTests(unittest.TestCase):
    def test_images_use_new_bundles_and_oci_referrers(self):
        package = (ROOT / '.github/workflows/package.yml').read_text()
        self.assertIn('cosign sign --yes --new-bundle-format --registry-referrers-mode=oci-1-1', package)
        for name in ('package.yml', 'release.yml'):
            workflow = (ROOT / '.github/workflows' / name).read_text()
            self.assertIn('cosign verify --new-bundle-format --registry-referrers-mode=oci-1-1', workflow)
            self.assertNotIn('--annotations', workflow)

    def test_approval_keeps_the_build_signature_and_reads_signed_image_labels(self):
        release = (ROOT / '.github/workflows/release.yml').read_text()
        self.assertNotIn('cosign sign', release)
        self.assertIn('packages: read', release)
        self.assertNotIn('id-token: write', release)
        self.assertIn('imagetools inspect "$IMAGE_REF"', release)
        self.assertIn('org.opencontainers.image.revision', release)
        self.assertIn('jq -n --arg identity "$build_identity"', release)
