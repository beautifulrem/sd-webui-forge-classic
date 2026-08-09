# Anima FreeFuse for Forge Neo

Native Anima port of [FreeFuse](https://github.com/yaoliliu/FreeFuse), pinned
for the design review at upstream commit
`7a76c3ac43ef3c27c7a3f8126e29e6021d837c80`.

This is a real two-pass implementation:

1. `Anima FreeFuse Euler` starts from the requested noise and runs only through
   the configured collection step with the selected subject LoRAs bypassed.
2. The configured Anima cross-attention block supplies concept attention;
   CrossAttn + SelfConcept top-k maps are converted into balanced, cleaned masks.
3. The sampler resets stateful guidance and restarts from the exact same initial
   noise. Each selected online LoRA's additive output is routed through its own
   mask, while an optional additive cross-attention bias suppresses wrong
   concept tokens and boosts the correct tokens.

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
- S-churn must remain zero so the collection and generation paths are
  deterministic and identical up to the collection step.
- Regional Conditioning and Artist Mixer are rejected in the same pass because
  they replace or multiply the conditioning stream from which FreeFuse derives
  token positions. Spectrum is forced to an actual model evaluation, while
  Modulation Guidance, LoRA layer weights, and the LoRA stage scheduler compose.
- Non-spatial Anima DiT LoRA paths (cross-attention context keys/values and
  timestep projections) remain unmasked in phase 2, matching FreeFuse's rule
  that only image-sequence LoRA outputs receive a spatial mask. Those DiT paths
  are bypassed during phase 1. Any separately loaded Anima text-encoder adapter
  has already produced the conditioning and is intentionally kept, so trigger
  identity remains available for concept-map collection.

The port supports standard additive Forge LoRA patches. It deliberately rejects
non-unit `strength_model` patches, which do not have an unambiguous independent
additive decomposition.
