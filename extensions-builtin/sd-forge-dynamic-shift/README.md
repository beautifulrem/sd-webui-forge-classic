# Dynamic Shift Scheduler (Forge Neo)

An extension for [sd-webui-forge-classic (neo branch)](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo) that ramps the flow-matching **shift** value across sampling steps instead of using one constant value. Built for flow models such as **Anima** (Cosmos-based), Lumina, Qwen-Image — anything where the built-in Shift slider is active.

## Why

The built-in Shift slider applies a single constant `s` to the whole sigma schedule via `sigma(t) = s·t / (1 + (s−1)·t)`. A **high** shift concentrates steps in the high-noise regime (better global composition / anatomy); a **low** shift concentrates them in the low-noise regime (better fine detail). A constant value forces one trade-off for the whole run. This extension interpolates the shift per-step — e.g. `5.0 → 1.5` gives you composition-heavy early steps *and* detail-heavy late steps in the same run.

## Install

Drop this folder into your webui's `extensions/` directory (or *Extensions → Install from URL* if you host it on a git repo), then restart.

## Usage

1. Open the **Dynamic Shift** accordion in txt2img or img2img.
2. Enable it, set **Shift @ first step** and **Shift @ last step**, pick a curve.
   The inline plot shows your dynamic schedule (orange) against the two constant-shift references (dashed/dotted).
3. Generate. The console logs the active ramp; parameters are written to infotext and restore correctly via *Send to txt2img* / PNG Info.

**Curves:** `Linear`, `Cosine` (eases in/out — the default, usually smoothest), `Exponential` (geometric interpolation, spends longer near the start value).

**Hires fix modes:**
- *Same as first pass* — reuse the same ramp for the hires pass.
- *Custom* — separate start/end values for the hires pass.
- *Static (built-in shift)* — hires pass uses a constant schedule at whatever the built-in Hires Shift slider is set to.

**XYZ grid:** three axes are registered — `[DynShift] Start`, `[DynShift] End`, `[DynShift] Curve`. Using any of them auto-enables the extension for that cell, so you can sweep ramps against a disabled baseline by including your UI values.

### Starting points for Anima

- 30–50 steps, CFG 4–5 as usual.
- `Shift 5.0 → 1.5, Cosine` — composition-first; good general default.
- `Shift 3.0 → 1.0, Linear` — subtle; close to stock shift 3 but with sharper late detail.
- `Shift 2.0 → 6.0` — reversed ramp; occasionally useful in img2img at low denoise where composition is already fixed.

## Notes / limitations

- While enabled, the **Schedule type** dropdown and the built-in **Shift** slider are bypassed for the sigma schedule (the extension takes over via `sampler_noise_scheduler_override`). Prompt-editing step percentages (`[a:b:0.5]`) still use the built-in shift for their timing — a minor, usually invisible interaction.
- Only activates on models with `use_shift = True`; it silently skips SDXL/SD1.5 and prints a console notice.
- If another script already sets a scheduler override, this extension defers to it.
- A constant ramp (start == end) reproduces the stock `FlowMatchEulerDiscrete` schedule exactly, which is handy for A/B sanity checks.

## How it works

`PredictionDiscreteFlow.timestep(sigma)` in Forge Neo is simply `sigma × 1000`, so shift only ever affects step *spacing*, never the sigma→timestep mapping. That means any strictly decreasing sigma sequence is a valid schedule. The extension builds one by evaluating the time-SNR shift formula with a per-step interpolated `s`, enforces monotonicity (needed for rising ramps), appends the terminal `0.0`, and hands it to the sampler through the standard `p.sampler_noise_scheduler_override(steps)` hook — no monkeypatching.
