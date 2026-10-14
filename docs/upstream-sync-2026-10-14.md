# Remi upstream sync — 2026-10-14

This sync merges Forge Neo `neo` at
[`46af880c`](https://github.com/Haoming02/sd-webui-forge-classic/commit/46af880c003d31846bbea984ce00fc7be247ce5b)
into `remi`. The previous baseline was `710f1e25`. Unlike the
[2026-10-02 selective port](upstream-sync-2026-10-02.md), this is a Git merge,
so `git log remi..upstream/neo` is empty afterwards and the next sync only has
to review newer commits. The four commits that the earlier port had already
adapted (`3a3dea7`, `ad88b6b`, `21886f4`, `ceff516`) resolve to their Remi
adaptations.

## Integrated as upstream

These commits apply unchanged, or only with whitespace-level conflict
resolution:

| Upstream commit | Change |
| --- | --- |
| `83391b9` Qwen-Image-2.1 | New model family: engine, DiT, VAE, Qwen3-VL 8B encoder, LoRA keys for the fused `img_mlp.gate_up` |
| `bd1d015` Anima Controllllite Mapping | `backend.nn.anima.BLOCK_MAPPINGS`, shared by the LoRA loader and the new 28→40/52 ControlLLLite block mapping |
| `8e07391` batch_cond_uncond | **Settings → Optimizations → Batch Cond/Uncond**. When disabled, cond and uncond run as separate model calls |
| `a607f87` offload, `97b26fb` mem | NeverOOM can always offload the UNet. `k_model` memory estimate |
| `558e26c` sdpa, `7aa43a9` mps, `ede60e1` flash, `58700b9` channels | SDPA fast path for small queries plus the gfx1151 contiguous-KV workaround; MPS INT_MAX slicing in VAE attention; causal flash attention; VAE channel fix |
| `c3b4291` mps, `46af880` api | API progress no longer forces a live preview. API script defaults come from `api_info` instead of re-running `ui()` |
| `ce6848b` folder, `aa144fa` refresh models, `af59a35` settings | VAE / Text Encoder dropdown can show subfolders. Refresh callbacks (Refiner, Hires, XYZ) via `main_entry.on_refresh_checkpoint`. Settings UI |
| `831d242` degrid | DeGrid post-processing in **Extras**; see the research note below |
| `dde0f32` load_torch_file | Only header errors are reported as a corrupt model |
| `d70373e` bump | comfy-kitchen 0.2.37 |
| `0b1783c`, `fef2647`, `89903ec`, `4f0ee8f`, `9c39688`, `899cf69`, `45f41b8`, `93c54c8`, `9cae3d6`, READMEs | Small fixes and cleanups in non-Anima engines, quantization, Refiner guard, scripts and UI |

## Adapted for Remi

| File | Resolution |
| --- | --- |
| `backend/diffusion_engine/anima.py` (`4636587` oops) | Upstream renamed `text_processing_engine_anima` to `text_processing_engine_qwen`. Bundled NegPiP and FreeFuse, and the external [forge-anima-3.8B](https://github.com/GumGum10/forge-anima-3.8B) extension (`anima3b/runtime.py`), read the old name. The engine stays under `text_processing_engine_anima`; `text_processing_engine_qwen` is a read/write alias. This closes the follow-up left open on 2026-10-02 |
| `backend/text_processing/_comfy.py` | Takes upstream's full file, because the rewritten non-Anima engines now import `SDTokenizer` and `EMBEDDINGS` from it. `SDClipModel.forward` keeps Remi's `memory_management.text_encoder_device()` instead of the embedding weight's device. Every pre-rewrite engine used that device; with offloaded (low-VRAM / NeverOOM) weights, the weight's device is the CPU |
| `backend/text_processing/anima_engine.py` | Keeps the 2026-10-02 Remi adaptation (Remi parser, raw tokenizers, `tokenize_line()` / `PromptChunk` for NegPiP and FreeFuse) |
| `extensions-builtin/sd_forge_spectrum/lib_spectrum/forecaster.py` (`90c2766` shape) | Keeps Remi's Spectrum. Its per-branch forecasters already rebuild on shape changes and handle cond/uncond split calls, which is the case upstream's shape guard covers (and that `batch_cond_uncond` off now makes common) |
| `backend/memory_management.py`, `backend/operations.py` | Equivalent hunks; upstream's form taken |
| `backend/operations_mixed_precision.py` | Dropped the duplicated `_GROUPED_INT8_FORMATS` (added on both sides) and the duplicated absolute import |
| `modules_forge/main_entry.py` | Neither `memory_management` nor `dynamic_args` is used after the merge; both imports dropped |
| `README.md`, `README.zh-CN.md` | Remi's READMEs kept; Qwen-Image-2.1 and ControlLLLite remapping added |
| `tests/test_anima_lora_support.py`, `tests/test_anima_lora_formats.py` | The `MAPPING_*` tables moved from `networks.py` to `backend.nn.anima.BLOCK_MAPPINGS`. Remi's `ANIMA_BLOCK_MAPPINGS` (which stays backend-free) is now checked against that table |

## Validation

See the merge commit message for the test run. ControlLLLite remapping,
Qwen-Image-2.1, and `batch_cond_uncond` off together with Anima Guidance, NAG,
FreeFuse, Regional and Spectrum still need a GPU generation check.
