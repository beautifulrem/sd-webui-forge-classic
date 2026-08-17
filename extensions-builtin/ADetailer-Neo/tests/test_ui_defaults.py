from __future__ import annotations

import sys
import unittest
from pathlib import Path


EXTENSION_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
for path in (REPOSITORY_ROOT, EXTENSION_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from lib_adetailer.ui import WebuiInfo  # noqa: E402


class WebuiInfoDefaultsTests(unittest.TestCase):
    def test_empty_checkpoint_install_has_a_safe_dropdown_default(self):
        info = WebuiInfo(
            ad_model_list=[],
            sampler_names=[],
            scheduler_names=[],
            t2i_button=None,
            i2i_button=None,
            checkpoints_list=[],
            vae_list=["None"],
        )

        self.assertIsNone(info.checkpoint_default)


if __name__ == "__main__":
    unittest.main()
