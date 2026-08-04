# Spectrum vNext for Anima

Forge Neo's built-in Spectrum panel forecasts the hidden feature immediately
before Anima's final DiT layer. A cached step still executes Anima's timestep
embedding, AdaLN final layer, unpatchify operation, and native flow-prediction
conversion. This avoids reusing a denoised output that belongs to an earlier
timestep.

Two refresh schedules are available:

- **Window** uses Spectrum's growing skip window.
- **SEA (auto-calibrated)** filters the current latent with SeaCache's
  spectral-evolution-aware metric. The first run of a matching
  sampler/CFG/resolution configuration measures a threshold; later runs load
  that threshold from `data/cache/spectrum-sea`.

`Conservative` is the default compatibility policy. It forecasts only simple
one-branch or standard two-branch Forge CFG batches, invalidates history when
the latent shape or text conditioning changes, and refuses unknown model
wrappers. `Strict` currently keeps every step actual because Forge does not
expose stable per-conditioning UUIDs. `Legacy / fastest` is an explicit
quality-risk opt-in for complex wrapper chains.

The minimum history is `Polynomial Degree + 2`. Increasing feature history can
raise VRAM usage substantially because Anima's pre-final hidden features are
large.
