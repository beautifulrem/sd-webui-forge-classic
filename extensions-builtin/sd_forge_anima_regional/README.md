# Anima Regional Conditioning

Built-in Forge Neo spatial prompt routing for Anima. Up to three regional
prompts are independently encoded, evaluated through selected cross-attention
blocks, and blended into normalized rectangular token masks. Coordinates use
the normalized `0..1` image frame and are recomputed for each base/Hires latent
resolution. Feather, base preservation, strength, block range, and sampling
window are exposed in the txt2img/img2img GUI.

Only conditional CFG rows are modified. Unconditional rows retain the normal
negative-prompt context. Overlapping regions are weight-normalized rather than
applied in order, so swapping region rows does not change the blend. Spectrum
is explicitly forced to actual forwards while regional routing is active.

The approach follows the independently encoded regional context and spatial
attention-routing design of `ComfyUI-AnimaRegionalConditioning`, adapted to
Forge's Anima attention implementation without persistent module monkeypatches.
