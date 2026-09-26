# Anima Cross-Attention Boost

Built-in, default-off port of the **front-loaded cross-attention boost**
(`--xattn_boost` of [anima_lora](https://github.com/sorryhyun/anima_lora),
MIT, commit `703b34b4`) for Anima. The plan of an image is written in the
first high-noise steps, the only window where Anima's cross-attention still
carries much text drive. While the model call's sigma is at or above the band,
every block's cross-attention residual is multiplied by the strength on the
positive-prompt rows only, so the boost lands in the CFG delta rather than in
the negative prompt. Norm matching then pulls each boosted image back toward
the mean token norm it would have had unboosted. It adds no model forwards.

| Control | Default | Notes |
| --- | --- | --- |
| Strength | `2.0` | Cross-attention residual gain; `1` is off. The reference ships `2.0`; `1.5` was weaker, `2.5`/`3` are untested. |
| Band (sigma) | `0.85` | Boosts model calls with sigma at or above it. `0.85` is the only band the reference tested (10 of 28 steps at its flow shift 3). The number of boosted steps depends on your scheduler and shift. |
| Norm matching | `0.5` | Exponent of the per-image mean-norm correction: `1` restores the unboosted mean token norm, `0` is the raw gain (the reference measured +18–25% saturation on complex prompts). |

What it does and does not do (reference findings, not measured in Remi):

- It helps relations and bindings between things the model knows
  (who holds what, multi-subject attributes, scene adherence). It does not
  create concepts the model barely knows.
- It amplifies every tag, wanted or not: framing (`cropped`, an artist's
  habitual close-up), counts (an extra person) and small identity details can
  drift. Artist/style tags in the prompt dilute the content gain.

Behavior:

- CFG's positive rows are boosted; negative rows run unchanged. At CFG 1
  there are only positive rows, and they are boosted.
- The gate is the absolute sigma of each model call. Hires. fix and img2img
  passes only get the boost when their denoise reaches the band. PC3 endpoint
  probes are gated by their own sigma like any model call.
- All blocks are boosted, on 2B, 2.9B and 3.8B alike (only 2B was tested by
  the reference). Reference-latent frames of edit models are included.
- Forwards are overridden only inside boosted model calls (block forwards
  for the call, attention forwards for each block call) and restored
  afterwards, also on errors. Spectrum runs actual forwards inside the band;
  per-block compile runs those steps eager. The gain and the per-image norm
  work at any latent size, so Anima SPEED keeps its half-resolution stage.
- The boost installs its cross-attention hook for each block call, on top of
  the hooks other features install per model call, so it scales the
  cross-attention output the block actually adds whatever order the scripts
  patch in. Composition with Regional Conditioning and FreeFuse is covered by
  CPU tests in both wrapper orders (Regional regions and FreeFuse's concept
  bias are boosted on the positive rows); NegPiP and NAG act inside the
  attention the hook wraps. Artist Mixer hooks the same way as Regional but
  has no composition test. None of these were tested together on a GPU.
- Norm matching rescales the state after cross-attention exactly as the
  reference does; the following MLP reads it through a LayerNorm that the
  scale does not change, so the correction is applied to the block output.

Evidence on Anima is the reference author's eyeball-adjudicated bench
(17–20 prompts, 1–3 seeds, 2B base model): the boost fixed 5 of 7
baseline-failed tags at strength 2 without the style collapse that raising
CFG in the same window caused. There are no quantitative metrics; see `NOTICE`.
