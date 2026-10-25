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
            self.assertIn('cosign verify --new-bundle-format', workflow)
            self.assertNotIn('cosign verify --new-bundle-format --registry-referrers-mode', workflow)
            self.assertNotIn('--annotations', workflow)
            self.assertIn("COSIGN_EXPERIMENTAL: '1'", workflow)

    def test_signing_options_are_checked_before_the_image_build(self):
        package = (ROOT / '.github/workflows/package.yml').read_text()
        self.assertLess(package.index('name: Check signing options before building'),
                        package.index('name: Build image'))
        self.assertIn('cosign sign --yes --new-bundle-format --registry-referrers-mode=oci-1-1 --help', package)
        self.assertIn('cosign verify --new-bundle-format --help', package)

    def test_approval_keeps_the_build_signature_and_reads_signed_image_labels(self):
        release = (ROOT / '.github/workflows/release.yml').read_text()
        self.assertNotIn('cosign sign', release)
        self.assertIn('packages: write', release)
        self.assertNotIn('id-token: write', release)
        self.assertIn('imagetools inspect "$IMAGE_REF"', release)
        self.assertIn('org.opencontainers.image.revision', release)
        self.assertIn('jq -n --arg identity "$build_identity"', release)

    def test_only_reviewed_verified_releases_move_latest(self):
        release = (ROOT / '.github/workflows/release.yml').read_text()
        package = (ROOT / '.github/workflows/package.yml').read_text()
        promotion = release.index('name: Promote the verified digest to latest')
        self.assertLess(release.index('test "$GPU_REVIEWED" = true'), promotion)
        self.assertLess(release.index('--output json "$IMAGE_REF"'), promotion)
        self.assertIn("if: github.ref == 'refs/heads/develop'", release)
        self.assertIn('environment: release', release)
        self.assertIn('cancel-in-progress: false', release)
        self.assertIn('imagetools create --prefer-index=false --tag "$latest" "$IMAGE_REF"', release)
        self.assertIn('test "$promoted" = "$DIGEST"', release)
        self.assertIn('--output json "$latest"', release)
        self.assertIn('backends-latest.yaml', release)
        self.assertNotIn(':latest', package)
        self.assertNotIn('imagetools create', package)
