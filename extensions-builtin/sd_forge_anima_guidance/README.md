# Anima Guidance & Corrections

Built-in Forge Neo controls for inference-time Anima guidance and correction
techniques. It includes **Skimmed CFG**, an anti-burn transform that limits
only CFG values identified as harmful, and **SMC-CFG**, an alpha-adaptive
sliding-mode controller operating in Anima's velocity space.

The panel is available in txt2img and img2img. It is inert for non-Anima
models, patches a cloned model for one generation, preserves an existing Forge
CFG function by wrapping it, and records active settings in generation info.

## Skimmed CFG

- `Skimming fallback CFG`: effective CFG used only for flagged values.
- `Full skim negative`: more strongly limits flagged negative-prompt values.
- `Disable flipping filter`: aggressive mode which disables the sign-flip
  safety predicate.
- Start/end/flip controls constrain the effect to a denoising interval.

The numerical predicates are adapted from
[Extraltodeus/Skimmed_CFG](https://github.com/Extraltodeus/Skimmed_CFG),
licensed under Apache-2.0. Modifications include a non-mutating numerical API,
Forge's residual-return CFG contract, existing-CFG-function composition, and
Anima-only lifecycle/UI integration. This derivative is distributed as part
of Forge Neo under the repository's AGPL-3.0 license; the upstream copyright
and Apache-2.0 notice are preserved here.

## SMC-CFG

SMC-CFG replaces the normal CFG combine with an adaptive sliding-mode
controller. The implementation converts Forge's denoised predictions to
velocity space, applies the controller, and returns the residual required by
Forge's CFG contract. `alpha=0.2` and `lambda=5.0` match the current Anima
defaults from
[ComfyUI-Spectrum-KSampler](https://github.com/sorryhyun/ComfyUI-Spectrum-KSampler).

SMC-CFG and another CFG-combine replacement cannot both own the same hook. If
SMC-CFG is explicitly selected, it replaces an existing CFG function and the
replacement is recorded in generation metadata. Skimmed CFG remains
composable and is applied to the cond/uncond predictions before SMC-CFG.

## CFG-Zero*

`CFG-Zero*` ([arXiv:2503.18886](https://arxiv.org/abs/2503.18886)) is a CFG
combine mode written from the paper. Per sample it projects the conditional
velocity onto the unconditional one, `s* = <v_cond, v_uncond> / ||v_uncond||^2`,
and guides with `v_uncond * s* + w * (v_cond - v_uncond * s*)`. Forge's CFG
hook residual is `x - denoised = sigma * v` for Anima's rectified flow and
`s*` ignores a common per-sample factor, so the combine runs on those
residuals directly. With `s* = 1` it is plain CFG; at CFG 1 it returns the
conditional prediction, so Forge still skips the negative pass there.

`Zero-init steps` (default 1, the paper's default) outputs zero velocity for
the first N sampler steps: `denoised = x`, so the solver leaves the latent
unchanged. A step is a distinct sigma, which is exact for single-evaluation
samplers; the extra sigmas of multi-evaluation samplers count as steps too.
PC3 probes do not consume steps. `0` keeps only the optimized scale.

Like SMC-CFG and FDG it replaces an existing CFG function (recorded in
metadata) and disables Momentum; Skimmed CFG, the CFG range and DCW compose.

## FDG (experimental)

Frequency-Decoupled Guidance splits the CFG update `cond - uncond` (in
denoised space) into a low band (2x2 average pool, bilinear upsample) and the
high-frequency remainder. The low band uses the normal CFG scale; the high
band uses `FDG detail guidance` (default 2.0). Equal scales reproduce plain
CFG. It is pure PyTorch, owns the CFG combine like SMC-CFG, and is disabled
with DCW (DCW wins only when it is the last switch changed), Momentum and the
`Anima ER SDE CNS` sampler.

## NAG

Normalized Attention Guidance works inside Anima's cross-attention rather than
on the CFG combine, so it also acts at CFG 1 (Turbo). For the positive rows it
extrapolates away from the negative prompt's attention output,
`z + scale * (z - z_neg)`, clamps the L1-norm growth to `tau`, and blends the
result back with `alpha` (defaults: scale 2.0, tau 2.5, alpha 0.5, active
over the first half of sampling, start 0.0 to end 0.5). Unconditional rows
and self-attention are untouched.

When the negative branch is in the same batch its K/V are reused. On
positive-only calls (CFG 1, or the positive half of a batch Forge split for
VRAM) the negative prompt's context is projected with the layer's K/V
weights, so no extra unconditional pass is needed. If neither is available
the generation records `Anima NAG: inactive: ...` in its metadata.
Regional/Artist Mixer synthetic branches are not guided.

## Momentum Guidance

Momentum keeps an EMA of the final (post-CFG) flow velocity across steps and
adds `strength * (v - EMA)` to the current velocity (defaults: strength 0.5,
EMA decay 0.6); the first step is unchanged. It adds no model forwards and
needs no negative pass at CFG 1. The state is per generation and PC3 probes do
not advance it. It is limited to single-evaluation Euler samplers (`Euler`,
`Anima Flow Euler`, `Anima FreeFuse Euler`) and requires the Standard CFG
mode.

## CFG active range

`Limit CFG to a sampling range` applies the CFG combine (including SMC-CFG,
CFG-Zero*, FDG and Skimmed CFG) only between `CFG range start` and `CFG range
end`, given as fractions of sampling progress (defaults 0.0 and 1.0).
Outside it the conditional prediction is used alone, and the negative forward
pass is skipped on those steps as long as no other hook still reads it.

## DCW

DCW applies the previous step's SNR-t bias correction to the mutable sampler
latent before the next denoiser evaluation. A post-CFG hook captures the prior
denoised prediction; Forge's pre-denoiser callback applies the correction only
when sigma advances, matching the upstream CALC_COND_BATCH timing without
monkeypatching the sampler.

The Anima default is `lambda=-0.015`, `one_minus_sigma`, and Haar `LL` only.
Other schedules and frequency-band masks are exposed in the GUI. DCW is a
post-step correction and composes with Standard CFG, SMC-CFG, and Skimmed CFG.

## Anima ER SDE CNS

`Anima ER SDE CNS` is a dedicated sampler entry which keeps Forge's ER-SDE
solver math and recolors only the stochastic noise draw at each step. The
recolorer redistributes a fixed variance budget across radial frequency bands
using the Anima-calibrated gamma matrix and RMS-normalizes every channel.

The GUI strength blends stock white noise with calibrated colored noise. CNS
does not affect Flow Euler, Flow UniPC2, or Flow PC3 because those ODE paths do
not contain ER-SDE's stochastic-noise seam. The pinned 6 KiB calibration file
is downloaded on first use over HTTPS, limited to 1 MiB, and verified against a
hard-coded SHA-256 before loading.

## Flow corrective sampler controls

The panel exposes the upstream parameters used by Forge's native `Anima Flow
UniPC2` and `Anima Flow PC3` paths. UniPC supports `bh1`/`bh2`, suppression of
early correctors, and optional per-sample dynamic thresholding. PC3 exposes
its maximum correction gamma and error tolerance. The defaults preserve the
conservative Diffusers-grid profiles; controls are written to generation
metadata only when their corresponding sampler is active.

## CLIP modulation guidance

The optional modulation path supplies the global CLIP-L conditioning that the
base Anima/Cosmos architecture was trained without. Conditional rows receive
`CLIP(base) + w * (CLIP(direction+) - CLIP(direction-))`; unconditional CFG
rows receive their own negative-prompt CLIP embedding without the direction.
The projected vector is multiplied by the learned per-block adapter scale and
added to each selected block's AdaLN-LoRA input. The final layer keeps the
native, unmodified timestep embedding.

Automatic mode downloads revision-pinned, SHA-256-verified weights only when
the control is enabled. Adapter `.pt` files are loaded with PyTorch
`weights_only=True`; CLIP uses a safetensors file and Forge's bundled FLUX
CLIP-L config/tokenizer. The large adapter projection runs once on CPU, and
only the small projected vectors and per-block scale rows are used during
sampling.

Spectrum composition is explicit: modulation runs around Spectrum, while
Spectrum forecasts features learned from actual modulated blocks. Hires passes
strip and rebuild both pass-scoped wrappers instead of nesting stale states.
