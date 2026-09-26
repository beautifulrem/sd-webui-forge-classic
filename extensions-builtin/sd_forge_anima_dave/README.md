# Anima DAVE Diversity

Built-in, default-off port of **DAVE** (DC Attenuation for diVersity
Enhancement, ICML 2026, [arXiv:2606.06813](https://arxiv.org/abs/2606.06813))
for Anima. Same-prompt images often share one layout across seeds because the
spatial mean (DC) of the DiT block features converges early. During the first
`tau` of the schedule DAVE removes `strength` of that mean from the selected
block outputs, `h ← h − strength · mean(h)`, and leaves the seed-specific
residual alone. It adds no model forwards.

| Control | Default | Notes |
| --- | --- | --- |
| Strength | `0.30` | Fraction of the DC removed per block. Not "more is stronger": layout diversity often peaks around `0.05–0.2`. |
| Tau | `0.10` | Leading fraction of the full schedule. Keep at or below `0.10`; wider windows garble text and hands far sooner than a higher strength does. |
| Blocks | `8-18` | 28-block (2B) indices. On 2.9B/3.8B each index maps to its original block, so the number of edited blocks stays at the tuned 11. |

Behavior:

- The mean is per sample and per channel over the generated image's tokens;
  reference-latent frames of edit models are neither averaged nor edited.
  CFG's conditional and unconditional rows are both edited, as in the reference.
- The window is measured on the full schedule: Hires. fix and img2img passes
  start part-way through it, so they only get DAVE when their denoise
  reaches the window (layout diversity belongs to the base pass).
- Block forwards are overridden only inside windowed model calls and restored
  afterwards, also on errors. Spectrum runs actual forwards inside the window;
  per-block compile runs those steps eager. The token mean works at any latent
  size, so Anima SPEED keeps its half-resolution stage. Regional, FreeFuse and
  Artist Mixer patch cross-attention, not the block output, and compose.
- PC3 endpoint probes are gated by their own sigma like any model call.

Ported from [ComfyUI-Anima-DAVE](https://github.com/sorryhyun/ComfyUI-Anima-DAVE)
(MIT, commit `83143e8d`), whose block band comes from the author's per-block
DC probe of Anima 2B. Quality evidence on Anima is qualitative (author seed
grids); see `NOTICE`.
