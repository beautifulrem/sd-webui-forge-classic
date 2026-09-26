# remi architecture notes

How the Anima features hook into Forge Neo, and the trade-offs that were made
on purpose. Read this before "fixing" one of them.

## Seams

| Seam | Where | Used by |
|---|---|---|
| Attention modifier chain | `backend/nn/anima_attention.py` (`ANIMA_ATTENTION_MODIFIERS`, `ANIMA_PROJECT_KV`) | NAG |
| Temporary forward overrides | `modules/forward_override.py` | Regional, Artist Mixer, FreeFuse, DAVE, cross-attention boost, NegPiP's DiT hook |
| `compute_qkv(value_context=...)` | `backend/nn/anima.py` | NegPiP, FreeFuse (V from the masked context) |
| `model_function_wrapper` chain | `call_model_wrapper` in `modules/anima_support.py` | Modulation, Regional, FreeFuse, Artist Mixer, DAVE, cross-attention boost, Spectrum |
| Spectrum "run a real forward" requests | `modules/spectrum_force.py` | every feature Spectrum cannot forecast |
| Stateful guidance registry | `modules/anima_guidance_state.py` | DCW, SMC-CFG, APG, CFG-Zero*, Momentum register; samplers that change what the history describes reset it (FreeFuse after its probe pass, SPEED at its resolution switch) |
| Active-feature registry | `set_feature_active` / `is_feature_active` in `modules/anima_feature_conflicts.py` | FreeFuse, Regional, Artist Mixer |
| Online-LoRA leases | `acquire_online_lora` / `release_online_lora` in `modules/anima_online_lora.py` (the last release restores the user's setting) | Stage scheduler, FreeFuse |
| Post-load validation | `Script.after_model_load` in `modules/scripts.py`, run by `process_images` once checkpoint/VAE overrides have loaded | features that check the loaded model or the final settings (Hires Guard) |
| Guaranteed teardown | `Script.cleanup` in `modules/scripts.py`, run from `process_images`' `finally` | features that must undo global state even when a run raises |
| Load-time quantization | `quantize_on_load` in `backend/nn/anima_quant.py`, via "Diffusion in Low Bits" | scaled FP8, INT8 ConvRot |

## Deliberate trade-offs

- **Upstream files stay close to upstream.** Changes to Forge Neo files
  (`backend/loader.py`, `modules/processing.py`, `modules/scripts.py`,
  `extensions-builtin/sd_forge_lora/networks.py`, ...) are kept to small hooks
  so upstream merges stay easy. For the same reason the per-block modulation
  locals in `Block.forward` (`backend/nn/anima.py`) mirror upstream line for
  line, and the 2B to 2.9B/3.8B block tables are duplicated in
  `modules/anima_lora_support.py` rather than edited out of `networks.py`; a
  test keeps the two identical.
- **Three LoRA extensions wrap `ModelPatcher.add_patches`** (layer weight,
  stage scheduler, FreeFuse metadata) through `backend/patcher/function_chain.py`
  instead of a patch-hook registry in `sd_forge_lora`. A registry would be
  cleaner but means rewriting ~5k lines of bundled extensions whose effect
  can only be checked on a GPU.
- **Regional, Artist Mixer and FreeFuse override cross-attention `forward`**
  rather than being attention modifiers. They swap in whole contexts (with
  their own NegPiP masks) and blend outputs; moving them onto the modifier
  chain is possible but changes numerics that need GPU A/B tests first.
- **NegPiP patches `compile_conditions`** to carry its mask through
  conditioning. This is the bundled Forge NegPiP extension's own design;
  changing it would mean diverging from that extension and upstream sampling.
- **Prompt rewrites are opt-in** (`modules/prompt_rewrites.py`): a global
  restore in `process_images` would also undo Loopback's intentional edits.
- **`ScriptAbort`** is only re-raised from `before_process_batch`, where the
  per-batch LoRA rules are validated; other hooks keep upstream's
  report-and-continue behaviour.
- **Not cached, on purpose:** FreeFuse's per-LoRA contributions, since the
  stage scheduler changes LoRA strengths every step through the same patch
  objects (plain LoRAs run as two low-rank matmuls each call; other adapters
  rebuild a dense delta); Spectrum's per-call prompt signature (split cond/uncond calls
  carry different branches within one step).
