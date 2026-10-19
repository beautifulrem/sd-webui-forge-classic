# Anima FreeFuse for Forge Neo

Native Anima port of [FreeFuse](https://github.com/yaoliliu/FreeFuse), pinned
for the design review at upstream commit
`7a76c3ac43ef3c27c7a3f8126e29e6021d837c80`.

This is a real two-pass implementation:

1. `Anima FreeFuse Euler` starts from the requested noise and runs only through
   the configured collection step with the selected subject LoRAs bypassed.
2. The configured Anima cross-attention block supplies concept attention;
   CrossAttn + SelfConcept top-k maps are converted into balanced, cleaned masks.
   SelfConcept compares the image tokens at unit RMS (the block's adaLN
   modulation would otherwise scale the dot products by ~60 and collapse every
   map onto one token), so the self-concept temperature does not depend on the
   model's activation scale.
3. The sampler resets stateful guidance and restarts from the exact same initial
   noise. Each selected online LoRA's additive output is routed through its own
   mask, while an optional additive cross-attention bias suppresses wrong
   concept tokens and boosts the correct tokens.

The collect block and the attention-bias blocks are written in the 2B
(28-block) layout. On the depth-expanded 40/52-block models the collect block
is the original block (not an inserted copy), and each attention-bias block
brings its inserted copies along, as the Artist Mixer's Blocks field does, so
the default `0-27` biases every block of any model (an older `0-51` still
selects them all).

Mask collection and routing preserve the image-batch dimension. Different
seeds/layouts in the same batch receive independent masks; masks are repeated
only across that image's CFG branches.

All algorithm parameters have controls in txt2img/img2img. Enabling the panel
selects the required sampler automatically and forces prompt LoRAs to reload in
Forge online mode. The original online-LoRA setting is restored afterwards.

## Requirements and safeguards

- An Anima checkpoint and two or three active subject LoRAs.
- Each GUI LoRA selector must match its loaded filename/stem.
- Each trigger phrase must occur verbatim in every positive prompt in a batch.
- Positive prompts must not contain `AND` (Forge's composable sub-prompts):
  the concept token positions and the one-mask-per-image collection only fit
  a single prompt. FreeFuse rejects such a prompt and aborts sampling rather
  than silently skipping; the console logs `Anima FreeFuse does not support
  AND prompts`.
- Prompt editing (`[a:b:N]`, `[a|b]`) is resolved per step like the main
  prompt, and the token positions are taken from every step's text. A prompt
  whose edits move the concept or background tokens between steps is
  rejected the same way; without a background phrase the background is the
  end-of-prompt slot, so any edit that changes the token count moves it.
- S-churn must remain zero so the collection and generation paths are
  deterministic and identical up to the collection step.
- Regional Conditioning and Artist Mixer are mutually exclusive because they
  replace or multiply the conditioning stream from which FreeFuse derives
  token positions. The GUI disables the other two when FreeFuse is selected;
  API/CLI processing gives FreeFuse priority and records the disabled features
  in infotext. Refiner checkpoint switching is cleared by the same resolver.
  Spectrum is forced to an actual model evaluation, while
  Modulation Guidance, LoRA layer weights, the LoRA stage scheduler and the
  Cross-Attention Boost compose. FreeFuse's cross-attention hook runs inside
  whatever forward the block has at call time (NegPiP's, the boost's), so the
  boost scales the biased cross-attention in either wrapper order.
- Non-spatial Anima DiT LoRA paths (cross-attention context keys/values and
  timestep projections) remain unmasked in phase 2, matching FreeFuse's rule
  that only image-sequence LoRA outputs receive a spatial mask. Those DiT paths
  are bypassed during phase 1. Any separately loaded Anima text-encoder adapter
  has already produced the conditioning and is intentionally kept, so trigger
  identity remains available for concept-map collection.

The port supports standard additive Forge LoRA patches. It deliberately rejects
non-unit `strength_model` patches, which do not have an unambiguous independent
additive decomposition.
