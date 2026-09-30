# remi architecture notes

How the Anima features hook into Forge Neo, and the trade-offs that were made
on purpose. Read this before "fixing" one of them.

## Seams

| Seam | Where | Used by |
|---|---|---|
| Attention modifier chain | `backend/nn/anima_attention.py` (`ANIMA_ATTENTION_MODIFIERS`, `ANIMA_PROJECT_KV`) | NAG |
| Masked attention (a call with a mask runs on PyTorch SDPA when the active backend is Sage, which ignores masks, or Flash, which rejects them) | `mask_honouring_attention` in `backend/nn/anima_attention.py` | FreeFuse's concept bias |
| Temporary forward overrides | `modules/forward_override.py` | Regional, Artist Mixer, FreeFuse, DAVE, cross-attention boost, NegPiP's DiT hook (`lib_negpip/anima.py`) and its SD1/SDXL cross-attention hook (`lib_negpip/sd.py`) |
| Per-call block forward pre-hooks (registered inside a `model_function_wrapper`, removed in its `finally`) | `_block_deltas` in `extensions-builtin/sd_forge_anima_guidance/lib_anima_guidance/modulation.py` | Modulation (adds its AdaLN-LoRA delta to the modulated blocks' input) |
| `compute_qkv(value_context=...)` / `compute_kv(value_context=...)` (cross-attention only) | `backend/nn/anima.py` | NegPiP, FreeFuse, NAG's negative-prompt K/V via `ANIMA_PROJECT_KV` (V from the masked context) |
| `model_function_wrapper` chain | `call_model_wrapper` in `modules/anima_support.py` | Modulation, Regional, FreeFuse, Artist Mixer, DAVE, cross-attention boost, CFG-Zero* zero-init (skips the forward of zero-init steps), Spectrum |
| Windowed forward-override patches (clone, gated `model_function_wrapper`, per-call overrides under a lock with guaranteed cleanup, wrapper flags, Spectrum request); `outermost` installs around the model function so the overrides wrap every other override of the same modules in either wrapper order | `windowed_override_patch` in `modules/anima_support.py` | Regional, DAVE (`outermost`: attenuates the cross-attention boost's norm-matched block output whichever hook runs first), cross-attention boost |
| Generated frames vs edit reference frames (an edit model's reference latents are appended after the generated frames along T, so their tokens follow the generated ones) | `generated_token_frames` / `pad_to_sequence` in `modules/anima_support.py` | Regional (region weight 0 on references), DAVE (mean over generated frames only), FreeFuse (LoRA routing factor 1 and no concept bias on references) |
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
- **Modulation uses per-call forward pre-hooks, not forward overrides.**
  Per-block compile runs any call with a forward override inside the blocks
  eager (`patched_inside` in `sd_forge_compile/anima_block_compile.py`), so
  an override would cost modulation users the compiled blocks for the whole
  generation. Dynamo traces a pre-hook as a frame of its own and keeps the
  compiled blocks, and adding or removing the hooks between calls
  recompiles nothing (`sd_forge_anima_guidance/tests/test_modulation_compile.py`). The hooks live
  only for the model call, like an override: between calls, and whenever
  modulation is off, the blocks carry none.
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
