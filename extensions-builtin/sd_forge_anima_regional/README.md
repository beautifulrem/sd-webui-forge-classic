# Anima Regional Conditioning

Built-in Forge Neo spatial prompt routing for Anima. Up to three regional
prompts are independently encoded, evaluated through selected cross-attention
blocks, and blended into normalized rectangular token masks. Coordinates use
the normalized `0..1` image frame and are recomputed for each base/Hires latent
resolution. Feather, base preservation, strength, block range, and sampling
window are exposed in the txt2img/img2img GUI. The block range is written in the
2B (28-block) layout, like FreeFuse and Artist Mixer: on 2.9B/3.8B each block
also selects its inserted copies, so a band covers the same depth of any model.
The sampling window (Start/End) is a fraction of model time (sigma through the
model's shift), not of the sampling steps. Region prompts support Forge prompt
editing (`[a:b:N]`, `[a|b]`), resolved per step like the main prompt.

Only conditional CFG rows are modified. Unconditional rows retain the normal
negative-prompt context. Overlapping regions are weight-normalized rather than
applied in order, so swapping region rows does not change the blend. A region's
strength is its blend opacity, capped at full replacement (less Base preserve):
above 1 it does not push harder inside the rectangle, it only brings the
feathered edge to full opacity sooner and gives the region more weight where it
overlaps weaker ones. Feather only softens edges inside the image: the ramp is
centred on the edge, so two regions that share an edge (the default left and
right halves) sum to full weight across the seam, and edges on the canvas
border are not faded. Spectrum
is explicitly forced to actual forwards while regional routing is active.

In the WebUI the regions are drawn over a preview shaped like the image (the
`region-boxes` widget, `webui/src/widgets/region-boxes`): drag a region to move
it, drag an edge or corner to resize it (edges snap to the image centre and to
the other regions; hold Shift to place freely), or focus it and use the arrow
keys (Shift resizes, Alt moves ten steps). Layout buttons tile the image into
two or three columns or rows. The X / Y / Width / Height / Active sliders stay
the stored parameters: the editor reads and writes them, so infotext, presets
and the API are unchanged. Its own script argument (`region_editor`, last) is
not read; argument lists without it are accepted.

The approach follows the independently encoded regional context and spatial
attention-routing design of [Comfyui-Anima-Regional-Conditioning](https://github.com/Sen-sou/Comfyui-Anima-Regional-Conditioning) (MIT, see NOTICE), adapted to
Forge's Anima attention implementation without persistent module monkeypatches.
