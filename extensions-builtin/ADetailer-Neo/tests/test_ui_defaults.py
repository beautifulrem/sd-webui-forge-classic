from __future__ import annotations

import sys
import unittest
from pathlib import Path

EXTENSION_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
# The WebUI parses sys.argv when modules.shared is first imported (as tests/conftest.py)
sys.argv = sys.argv[:1]
for path in (REPOSITORY_ROOT / "modules_forge" / "packages", REPOSITORY_ROOT, EXTENSION_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from lib_adetailer.ui import WebuiInfo  # noqa: E402


class WebuiInfoDefaultsTests(unittest.TestCase):
    def info(self, checkpoints):
        return WebuiInfo(
            ad_model_list=list,
            sampler_names=list,
            scheduler_names=list,
            checkpoints_list=lambda: checkpoints,
            vae_list=lambda: ["None"],
        )

    def test_empty_checkpoint_install_has_a_safe_dropdown_default(self):
        self.assertIsNone(self.info([]).checkpoint_default)

    def test_checkpoint_default_follows_the_current_list(self):
        checkpoints = ["a.safetensors"]
        info = self.info(checkpoints)
        self.assertEqual(info.checkpoint_default, "a.safetensors")
        checkpoints.insert(0, "b.safetensors")
        self.assertEqual(info.checkpoint_default, "b.safetensors")


if __name__ == "__main__":
    unittest.main()
