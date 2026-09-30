from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from modules import sd_samplers_anima_cns
from modules.sd_samplers_anima_cns import CNSRecolorer, radial_bins


class CNSTests(unittest.TestCase):
    def _recolorer(self, strength):
        gamma = np.full((1, 2, 8), 0.25, dtype=np.float64)
        aspects = np.asarray([[1024, 1024]], dtype=np.float64)
        sigmas = np.asarray([1.0, 0.5, 0.0], dtype=np.float64)
        return CNSRecolorer(gamma, aspects, sigmas, strength)

    def test_radial_bins_are_bounded(self):
        bins = radial_bins(8, 12, 16)
        self.assertEqual(bins.shape, (8, 12))
        self.assertGreaterEqual(int(bins.min()), 0)
        self.assertLess(int(bins.max()), 16)

    def test_zero_strength_is_exact_passthrough(self):
        white = torch.randn(2, 4, 8, 8)
        result = self._recolorer(0.0).recolor(white, 0.5)
        self.assertIs(result, white)

    def test_recolored_noise_is_finite_and_rms_normalized(self):
        torch.manual_seed(8)
        white = torch.randn(2, 4, 16, 16)
        result = self._recolorer(1.0).recolor(white, 0.5)
        self.assertTrue(torch.isfinite(result).all())
        std = result.float().std(dim=(-2, -1))
        self.assertTrue(torch.allclose(std, torch.ones_like(std), atol=1e-4, rtol=1e-4))

    def test_device_gamma_lookup_matches_numpy_interp(self):
        rng = np.random.default_rng(3)
        gamma = rng.uniform(-0.2, 1.1, size=(3, 12, 16))
        aspects = np.asarray([[1024, 1024], [832, 1216], [1216, 832]], dtype=np.float64)
        sigmas = np.concatenate([np.sort(rng.uniform(0.0, 1.0, 12))[::-1], [0.0]])
        recolorer = CNSRecolorer(gamma, aspects, sigmas, 1.0)
        recolorer._select_aspect(104, 152)
        midpoints = recolorer.sigmas
        probes = [-0.5, 0.0, midpoints[0], midpoints[5], midpoints[-1], 1.0, 2.0, *rng.uniform(0.0, 1.0, 20)]
        for sigma in probes:
            expected = np.sqrt(np.clip(1.0 - np.asarray([np.interp(sigma, midpoints, recolorer.selected_gamma[:, index]) for index in range(16)]), 0.0, 1.0))
            for value in (sigma, torch.tensor(sigma, dtype=torch.float32), torch.tensor(sigma, dtype=torch.float64)):
                actual = (1.0 - recolorer._gamma_row(value, torch.device("cpu"))).clamp(0.0, 1.0).sqrt()
                np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-5, atol=1e-6, err_msg=f"sigma={sigma}")

    def test_single_calibrated_sigma_is_constant_like_numpy(self):
        recolorer = CNSRecolorer(np.linspace(0.0, 0.5, 8).reshape(1, 1, 8), np.asarray([[1, 1]]), np.asarray([0.7, 0.0]), 1.0)
        recolorer._select_aspect(8, 8)
        for sigma in (0.1, 0.7, 3.0):
            np.testing.assert_allclose(recolorer._gamma_row(sigma, torch.device("cpu")).numpy(), np.linspace(0.0, 0.5, 8), atol=1e-7)

    def test_a_failed_prefetch_is_not_retried_every_run(self):
        process = SimpleNamespace(extra_generation_params={})
        failing = mock.Mock(side_effect=OSError("offline"))
        with mock.patch.object(sd_samplers_anima_cns, "_GAMMA_ARRAYS", None), mock.patch.object(
            sd_samplers_anima_cns, "_GAMMA_FAILED_AT", None
        ), mock.patch.object(sd_samplers_anima_cns, "ensure_gamma_file", failing):
            self.assertFalse(sd_samplers_anima_cns.prefetch_gamma(process))
            self.assertFalse(sd_samplers_anima_cns.prefetch_gamma(SimpleNamespace(extra_generation_params={})))
        self.assertEqual(failing.call_count, 1)
        self.assertEqual(process.extra_generation_params["Anima CNS"], "white-noise fallback: gamma file unavailable (OSError)")

    def test_the_sampler_never_downloads_it_reads_the_prefetched_file(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "anima_cns" / "cns_gamma.npz"
            download = mock.Mock(side_effect=AssertionError("downloaded inside the sampler"))
            with mock.patch.object(sd_samplers_anima_cns, "_GAMMA_ARRAYS", None), mock.patch.object(
                sd_samplers_anima_cns, "_GAMMA_FAILED_AT", None
            ), mock.patch.object(sd_samplers_anima_cns, "_gamma_path", lambda: target), mock.patch.object(
                sd_samplers_anima_cns, "ensure_pinned_download", download
            ):
                # Not prefetched: white noise, recorded, and no download.
                process = SimpleNamespace(extra_generation_params={})
                self.assertIsNone(CNSRecolorer.try_calibrated(1.0, process))
                self.assertIn("FileNotFoundError", process.extra_generation_params["Anima CNS"])
                download.assert_not_called()

                # The prefetch (before sampling) downloads; the sampler then reads it.
                def fetch(**kwargs):
                    target.parent.mkdir(parents=True)
                    np.savez(target, gamma=np.full((1, 2, 8), 0.25), aspects=np.asarray([[1, 1]]), sigmas=np.asarray([1.0, 0.5, 0.0]))
                    return str(target)

                download.side_effect = fetch
                self.assertTrue(sd_samplers_anima_cns.prefetch_gamma(SimpleNamespace(extra_generation_params={})))
                self.assertEqual(download.call_count, 1)
                download.side_effect = AssertionError("downloaded inside the sampler")
                self.assertEqual(CNSRecolorer.try_calibrated(0.5, None).strength, 0.5)

                # A later process (tables not in memory) reads the verified file from disk.
                sd_samplers_anima_cns._GAMMA_ARRAYS = None
                with mock.patch.object(sd_samplers_anima_cns, "GAMMA_SHA256", sd_samplers_anima_cns.sha256_file(str(target))):
                    self.assertIsNotNone(CNSRecolorer.try_calibrated(1.0, None))
                self.assertEqual(download.call_count, 1)

    def test_a_tampered_gamma_file_is_not_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "cns_gamma.npz"
            target.write_bytes(b"not the pinned table")
            with mock.patch.object(sd_samplers_anima_cns, "_GAMMA_ARRAYS", None), mock.patch.object(
                sd_samplers_anima_cns, "_gamma_path", lambda: target
            ):
                process = SimpleNamespace(extra_generation_params={})
                self.assertIsNone(CNSRecolorer.try_calibrated(1.0, process))
        self.assertIn("RuntimeError", process.extra_generation_params["Anima CNS"])

    def test_the_prefetchs_reason_is_kept_over_the_samplers(self):
        process = SimpleNamespace(extra_generation_params={})
        with mock.patch.object(sd_samplers_anima_cns, "_GAMMA_ARRAYS", None), mock.patch.object(
            sd_samplers_anima_cns, "_GAMMA_FAILED_AT", None
        ), mock.patch.object(sd_samplers_anima_cns, "ensure_gamma_file", mock.Mock(side_effect=TimeoutError())), mock.patch.object(
            sd_samplers_anima_cns, "downloaded_gamma_file", mock.Mock(side_effect=FileNotFoundError())
        ), mock.patch.object(sd_samplers_anima_cns.logger, "warning") as warning:
            sd_samplers_anima_cns.prefetch_gamma(process)
            self.assertIsNone(CNSRecolorer.try_calibrated(1.0, process))
        self.assertEqual(process.extra_generation_params["Anima CNS"], "white-noise fallback: gamma file unavailable (TimeoutError)")
        self.assertEqual(warning.call_count, 1)

if __name__ == "__main__":
    unittest.main()
