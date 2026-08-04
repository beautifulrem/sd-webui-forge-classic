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
