# Anima Prompt-Schedule Rescaler

A small companion extension for **Forge Classic (neo)** that keeps
[prompt-editing / scheduling](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#prompt-editing)
schedules behaving the way you intended when a prompt moves from a txt2img pass
to an img2img pass.

## The problem it solves

A schedule like `[@guweiz:@wlop:15]` written for a **35-step txt2img** means
"switch style at schedule-step 15" — about 43% of the way through the run.

When you send that prompt to **img2img at 35 steps and 0.5 denoise**, the timing
gets thrown off. Here is what actually happens inside Forge (verified against
the neo source, not assumed):

1. The conditioning schedule is built against the **full Steps slider value**
   (`setup_conds` passes `p.steps` = 35) — img2img does *not* rebuild it for the
   shorter executed run. So `15` still means schedule-step 15 of 35.
2. img2img only runs the tail: `t_enc = int(min(denoise, 0.999) × steps)` —
   here `int(0.5 × 35) = 17` steps.
3. The denoiser's step counter starts at **0** and counts up per step with **no
   offset** for the denoise window (`sd_samplers_cfg_denoiser.py`:
   `reconstruct_cond_batch(cond, self.step)`).

So the switch fires when that 0-based counter hits 15 — the **16th of 17
executed steps**, ~88% through the refine, when you meant it at ~43%. And any
`when` past `t_enc` (17) never fires at all. That late/never timing is what
"throws off the style."

## What it does

For each scheduled `when` it works out the fraction `f` of the original run the
switch sat at, then re-targets it to the schedule-step that fires at the **same
fraction of the executed img2img pass**:

```
f      = when / source_steps        (or the value itself if it was a fraction)
t_enc  = int(min(denoise, 0.999) × target_steps)   # steps img2img runs
when_new = round(f × t_enc)                          # 0-based step that fires there
```

So `[@guweiz:@wlop:15]` at source 35 → img2img 35 steps / 0.5 denoise becomes
`[@guweiz:@wlop:7]` (`round(0.43 × 17) = 7`). The `when` goes **down**, because
the refine only runs ~17 steps and you want the switch ~43% of the way into
*those* steps. With `denoise = 1` and equal step counts it is a no-op up to
rounding.

Alternation blocks (`[a|b|c]`), attention weights (`(detail:1.3)`), and plain
text are never touched. Nested schedules are handled at every depth.

## Two ways to use it

Both live in the **Anima Prompt-Schedule Rescaler** accordion (txt2img and
img2img), and both are off/manual by default.

### Manual tool
Paste a prompt, set the **source steps** (what it was authored for), the
**target steps**, and the **denoise** of the pass you're about to run, then hit
**Rescale schedule**. Copy the result into your prompt box. Tick *Output as
fractions* if you'd rather read the result as a decimal (it maps to the same
firing step).

### Auto-apply (img2img only)
Tick **Auto-rescale this img2img generation's schedules** and set **Source
steps** to your original txt2img step count. On generate, the prompt and
negative prompt schedules are rewritten for that single run using the live
**Steps** slider and **Denoising strength**. It records what it did in the PNG
info (`Anima schedule rescale: src35->tgt35@d0.5`).

This path is wrapped so any parsing problem falls back to your original prompt —
it can't break a generation.

Forge Neo's **img2img fix steps** option is detected automatically. When that
option is enabled, the rescaler targets the full requested step count instead
of multiplying it by denoise, matching Neo's alternate `setup_img2img_steps`
branch.

## Notes / caveats

* The math targets *nominal* steps. Predictor-corrector solvers (PC3, UniPC,
  Heun, AB2) call the model more than once per integration step, and Forge's
  step counter counts model calls, so the exact firing point can drift slightly
  for those solvers. The first-order fix (`round(f × t_enc)`) is still the right
  target; nudge by a step if you want a switch slightly earlier/later.
* It only rewrites *scheduled* blocks (`[from:to:when]`, `[to:when]`,
  `[from::when]`). Per-step alternation has no `when` and needs no rescaling.
* This extension is independent of the Anima sampler — it works with any
  sampler — but is designed to pair with it.

## Install

Drop the `sd-forge-anima-prompt-rescale` folder into your Forge `extensions/`
directory and restart. No extra dependencies.
