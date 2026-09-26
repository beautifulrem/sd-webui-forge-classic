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
The switching gain `alpha * mean|e|` is taken per image (the reference averages
over the whole batch, which is the same at batch size 1), so images in a batch
do not affect each other.

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
Those steps skip the model forward entirely (a model wrapper returns the
input as the denoised prediction), and Spectrum runs them as forced actual
steps so it does not learn from them.
PC3 probes do not consume steps. `0` keeps only the optimized scale.
Zero-init is meant for the start of sampling from pure noise: it applies only
to a pass whose first sigma is the schedule's maximum, so hires and img2img
passes (which start mid-schedule) are not zeroed. It is the outermost CFG
hook, so with the CFG range it still zeroes the first steps of the schedule,
not the first steps inside the range.

Like SMC-CFG, APG and FDG it replaces an existing CFG function (recorded in
metadata) and disables Momentum; Skimmed CFG, the CFG range and DCW compose.

## APG

Adaptive Projected Guidance ([arXiv:2410.02416](https://arxiv.org/abs/2410.02416))
follows the paper's Algorithm 1 (also diffusers'
`guiders/adaptive_projected_guidance.py`, Apache-2.0). The CFG update
`cond - uncond` is accumulated with momentum (`running = diff + momentum *
running`), capped to `APG norm threshold` in L2 norm per image, and split into
the parts parallel and orthogonal to the conditional prediction; the result is
`cond + (w - 1) * (orthogonal + eta * parallel)`. It works on Forge's denoised
(x0) predictions because the paper applies the projection to the denoised
prediction (the paper finds the projection has little effect otherwise). `eta 1`, no cap
and momentum 0 are plain CFG. Defaults: eta 0, norm threshold 0 (off; the
norm covers the whole latent, so the useful value grows with resolution),
momentum -0.5 (the paper's reverse momentum). The momentum is per pass, reset
after FreeFuse's probe pass, and not advanced by PC3 probes; at CFG 1 APG
returns the conditional prediction. Like the other combine modes it replaces
an existing CFG function and disables Momentum Guidance.

## FDG (experimental)

Frequency-Decoupled Guidance splits the CFG update `cond - uncond` (in
denoised space) into a low band (2x2 average pool, bilinear upsample) and the
high-frequency remainder. The low band uses the normal CFG scale; the high
band uses `FDG detail guidance` (default 12.0, the reference's
`strength_high`), an absolute scale like CFG. The paper guides the high band
harder than the low band, so keep it above CFG; below CFG it weakens detail
guidance. Equal scales reproduce plain CFG. It is pure PyTorch, owns the CFG combine like SMC-CFG, and is disabled
with DCW (DCW wins only when it is the last switch changed), Momentum and the
`Anima ER SDE CNS` sampler.

## NAG

Normalized Attention Guidance works inside Anima's cross-attention rather than
on the CFG combine, so it also acts at CFG 1 (Turbo). For the positive rows it
extrapolates away from the negative prompt's attention output,
`z + scale * (z - z_neg)`, clamps the L1-norm growth to `tau`, and blends the
result back with `alpha` (defaults: scale 2.0, tau 2.5, alpha 0.5, active
over the first half of sampling, start 0.0 to end 0.5). Unconditional rows
and self-attention are untouched. Outside its window NAG reports itself
inactive (`active(transformer_options)`), so torch.compile keeps the compiled
blocks on those steps.

When the negative branch is in the same batch its K/V are reused. On
positive-only calls (CFG 1, or the positive half of a batch Forge split for
VRAM) the negative prompt's context is projected with the layer's K/V
weights (with NegPiP's V-only negation of `(tag:-1)` negative-prompt tokens,
as in the batched branch), so no extra unconditional pass is needed. If neither is available
the generation records `Anima NAG: inactive: ...` in its metadata.
Regional/Artist Mixer synthetic branches are not guided.

## Momentum Guidance

Momentum keeps an EMA of the final (post-CFG) flow velocity across steps and
adds `strength * (v - EMA)` to the current velocity, using the EMA of the
previous steps, then folds `v` into the EMA as in the reference
(`pamparamm/sd-perturbed-attention` `mg_nodes.py`; defaults: strength 0.5,
EMA decay 0.6); the first step is unchanged. It adds no model forwards and
needs no negative pass at CFG 1. The state is per generation and PC3 probes do
not advance it. It is limited to single-evaluation Euler samplers (`Euler`,
`Anima Flow Euler`, `Anima FreeFuse Euler`) and requires the Standard CFG
mode.

## CFG active range

`Limit CFG to a sampling range` applies the CFG combine (including SMC-CFG,
CFG-Zero*, APG, FDG and Skimmed CFG) only between `CFG range start` and `CFG range
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
post-step correction and composes with every CFG mode except FDG, and with
Skimmed CFG.

DCW needs one main model evaluation per step: Euler, Euler a, DPM++ 2M,
DPM++ 2M/3M SDE, ER SDE (and CNS), LMS, LCM, Res Multistep, the Anima Flow
samplers (PC3's endpoint probe is skipped), Anima SPEED and Anima FreeFuse
Euler. On other samplers (Heun, DPM2, DPM++ SDE/2S, Restart, UniPC, DDIM, ...)
the correction would land on an intermediate evaluation, so DCW is disabled
and the reason is recorded in `Anima conflict resolution`. The same happens
with an inpaint mask: the denoiser then evaluates a blended copy of the
latent, so the correction could not reach the sampler's latent.

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
base Anima/Cosmos architecture was trained without. With `proj` the adapter's
Linear -> SiLU -> Linear projection of a CLIP-L pooled vector, conditional rows
receive `proj(base) + w * (proj(direction+) - proj(direction-))` (the direction
is taken after projection); unconditional CFG rows receive the projection of
their own negative-prompt CLIP embedding without the direction. The projected
vector is multiplied by the learned per-block adapter scale and
added to each selected block's AdaLN-LoRA input. The final layer keeps the
native, unmodified timestep embedding.

The pinned adapter's scales cover Anima 2B's 28 blocks. On the
depth-expanded 2.9B (40 blocks) and 3.8B (52 blocks) models each scale row
goes to its preserved original block and the inserted blocks get no
modulation, following ComfyUI-Spectrum-KSampler's `mod_guidance_layouts.py`
(MIT, commit `b1ad4f9`) with Forge's Anima block lineage. `Start block` /
`End block` count the adapter's blocks, so they select the same original
blocks on every depth. An unsupported model depth is rejected before
anything is downloaded, and the adapter's shape is checked against the model
before CLIP-L is fetched.

CLIP sees plain text, as in the reference (whose ComfyUI tokenizer drops
emphasis syntax): prompt editing is resolved to the pass's first-step prompt
(one pooled vector serves the whole pass), `(text:1.2)` / `[text]` weights
are dropped and `BREAK` becomes a space.

Automatic mode downloads revision-pinned, SHA-256-verified weights only when
the control is enabled. Adapter `.pt` files are loaded with PyTorch
`weights_only=True`; CLIP uses a safetensors file and Forge's bundled FLUX
CLIP-L config/tokenizer. The large adapter projection runs once on CPU, and
only the small projected vectors and per-block scale rows are used during
sampling.

Spectrum composition is explicit: Spectrum's model wrapper wraps the
modulation wrapper. Actual steps call through it, so the modulated blocks run
and Spectrum learns their features; cached steps skip the blocks and forecast
those modulated features, then run the unmodulated final layer. Every pass
(base, Hires) starts from Forge's post-LoRA model copy, so both wrappers are
installed fresh per pass and never nest stale states.
