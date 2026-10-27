# Remi upstream sync — 2026-10-27

This sync merges Forge Neo `neo` at
[`c6c2b43d`](https://github.com/Haoming02/sd-webui-forge-classic/commit/c6c2b43d03580160db7acb3c3bc22284a1e097e6)
into `remi`. The previous merge base was `46af880c`
([2026-10-14](upstream-sync-2026-10-14.md)). It is the first sync after the
UI rewrite, so it follows [upstream-absorption.md](remi-webui/upstream-absorption.md):
the runtime is a Git merge (`78f325d3`), and the changes to upstream's
Gradio UI are translated into `params()`, settings widgets and SPA code.
All 26 upstream commits are classified below.

Notable for users:

- **New model families**: Moss-Image and Moss-Image-Edit (OpenMOSS), with the
  "Moss" UI preset, Skip Layer Guidance and the new **JSON Builder** page for
  their structured prompts.
- **Extras → DeGrid is now off by default**, as upstream. Enable it in the
  Extras form (or with `"degrid": {"degrid_enabled": true}`). An `/sdapi`
  extras call without script arguments no longer runs it.
- txt2img Width / Height go up to **4096**; img2img stays at 2048.
- **torch 2.14.0+cu130** / torchvision 0.29.0 on Linux and Windows (the
  separate Nunchaku pair is gone). The startup check warns below 2.14.0.
- **Pillow 13.0.0** (major bump), comfy-kitchen 0.2.40, accelerate 1.15.0,
  GitPython 3.1.62, pillow-heif 1.8.1, tqdm 4.70.1.
- **`--onnxruntime-gpu` was removed** upstream (`backend/args.py`). A launch
  command or `COMMANDLINE_ARGS` that still passes it now stops with an
  argparse error: remove the flag.

## Integrated as upstream

Merged unchanged, or with conflicts that only kept Remi's hooks:

| Upstream commit | Change |
| --- | --- |
| `be6fa5d4` Moss-Image, `c6c2b43d` Moss-Image-Edit | Moss-Image family: engine, DiT, Qwen3.5-9B text encoder (`backend/nn/llm/llama.py`), tokenizer (`.xz`, decompressed into a gitignored path), MOSS-Image-SFT configs, `huggingface_guess` detection, `PresetArch.moss` (Euler / Simple / 50 steps / CFG 3 / shift 5); Moss-Image-Edit reference latents (`dynamic_args.can_reference`) and the `moss_do_reference` setting with its on-change handler |
| `be6fa5d4` + `a83c1c82` unpatch | Skip Layer Guidance, `modules/processing_scripts/slg.py` (new, settings only, no `ui()`): between the start and end percent the uncond call of a Moss-Image step skips the listed `single_block`s; the patch of `calc_cond_uncond_batch` is a pass-through for other models and reverted on unload. It loads in Remi's runner without `params()` (nothing to port); its infotext keys paste back as setting overrides (`tests/test_processing_script_slg.py`) |
| `0c388f8b` lora | Moss LoRA keys (bare module paths, fused `qkv` split into q / k / v) in `comfy/lora.py` |
| `d880e8b8` oops | Emphasis base returns `z` |
| `6b3a78d7` type | LoRA te multiplier with a non-string positional |
| `e78e1453` attn | VAE attention uses `pytorch_attention_enabled_vae()`; AMD no longer excluded from PyTorch attention |
| `9d274922` gguf | No fp16 linear fast path for GGUF weights |
| `e279f808` mixed, `f7f23149` shush | Mixed-precision layout name in the logs, `k_model` memory for mixed ops; `comfy_attention` keys no longer reported as unexpected |
| `796c65f7` cpu | No fp8 / int8 / nvfp4 / mxfp8 compute on a CPU device (`tests/test_quant_platform_policy.py` follows) |
| `2c24de16` alpha | **Settings → Saving Images/Grids → Alpha Threshold** (`ui.Slider`, default 16; 0 disables): saving an RGBA image zeroes alpha below it and blackens fully transparent pixels. The gallery's save goes through the same `images.save_image_with_geninfo` (`tests/test_images_alpha_threshold.py`) |
| `ad607938` xyz (`modules/processing.py`) | VAE override through `forge_additional_modules` |
| `327b7f89`, `10bb5ed9` bump | comfy-kitchen 0.2.38 → 0.2.40 |
| `dc014551`, `c358e55d` update README | Forge Neo version 2.31 (`forge_version.py`); the README model list goes into Remi's READMEs (en, zh-CN) |

## Translated into the Remi UI

| Upstream commit / file | Remi change |
| --- | --- |
| `be6fa5d4` `extensions-builtin/sd_forge_json_builder` (new, JavaScript only: an empty `on_ui_tabs` tab plus `main.js`, `compile.js`, `decompile.js`, `style.css`) | `scripts/builder.py` registers the `json-builder` page (`remi.pages`, `widget="json-builder"`, Gradio-free); the editor is `webui/src/pages/custom/json-builder/` (canvas, draft, `compile.ts` / `decompile.ts` following upstream's rules and messages, Send to txt2img / img2img through `features/send-to`). A canvas larger than the target form accepts is still sent, with a warning toast (img2img stays at 2048; upstream sends it silently). The JavaScript and CSS are deleted again; no legacy snapshot (no `Script.ui()`). See [ports/json-builder.md](remi-webui/ports/json-builder.md) and the inventory row; the page is in the mock backend and the axe / flow e2e runs |
| `be6fa5d4` `modules/ui.py` txt2img size | `remi/core/forms.py` `_dimensions`: txt2img Width / Height maximum 4096, img2img 2048; tab schema fixtures and the mock forms follow |
| `3a0e725a` off (`scripts/degrid.py`) | `params()`: `ui.Toggle("degrid_enabled", False)`, section closed, as upstream's `InputAccordion(False, …)`; `process(…, degrid_enabled=True)` unchanged, as upstream. The legacy contract test lists the default as the one expected change to `degrid_extras.json` (not re-recorded); `test_degrid_is_off_by_default` |
| `04772d93` edit (`image_stitch.py` help text) | `T2I_INFO` / `I2I_INFO` (Markdown) say "For **Edit Models** (Flux.1-Kontext, Flux.2-Klein, Qwen-Image-Edit, Qwen-Image-2.1, Anima Edit, Krea2 Edit)"; the list is inline because Markdown has no hover title; both zh_CN entries re-keyed |
| `93950eb8` custom (`modules_forge/presets.py`) | The preset sampler / scheduler settings are `ui.Choice(allow_custom=True)` (upstream's `allow_custom_value`) |
| settings in `be6fa5d4`, `2c24de16`, `f74f60c5`, `c6c2b43d` | `alpha_threshold` (`ui.Slider`), `slg_enable` (`ui.Toggle`), `slg_single_layers` (`ui.Text`), `slg_start_percent` / `slg_end_percent` (`ui.Number`), `moss_do_reference` (`ui.Choice`), `xyz_use_cost`; zh_CN labels; the "MOSS" UI-preset settings section. Upstream's `prompt_box_style` stays out (Gradio only) |
| `be6fa5d4`, `c6c2b43d` model lists | `remi/network_metadata.py` `ARCHS` and the `moss[-_ ]?image` name pattern; the SPA model bar label "Moss-Image" and the mock preset list |

## Adapted for Remi

| File | Resolution |
| --- | --- |
| `extensions-builtin/sd_forge_spectrum/` (`34022261` rewrite Spectrum; forked, §1.1) | Remi's fork kept (`tools/upstream_forks.py restore`; upstream's `lib_spectrum/__init__.py` with `SpectrumNode` stays out). Ported: the removal of the "Spectrum does not support Ignore/Skip Negative Prompt" early return; Remi's per-branch history drops the uncond forecaster for the steps without it and keeps caching cond (`extensions-builtin/sd_forge_spectrum/tests/test_skipped_uncond.py`). New: `_replaced_blocks` puts the block keys of `transformer_options["patches_replace"]` into each branch's signature, so a branch whose blocks Skip Layer Guidance replaces for part of the pass is rebuilt instead of forecast across the switch (`extensions-builtin/sd_forge_spectrum/tests/test_replaced_blocks.py`; seam documented in [remi-architecture-notes.md](remi-architecture-notes.md#seams) and §1.1). Already covered by Remi: per-branch / per-item forecasters, `ready()` with `M + 2` samples, new-pass detection on a rising sigma, the hires pass's own step count (`pass_context.py`). Left out: the `nn.Module` structure, the fixed history `K = 100` (Remi's "Feature History" control), bf16 / fp16 coefficients and the separate CUDA stream (Remi solves in float64 on the CPU) |
| `modules/launch_utils.py` (`a02a0516` torch) | Upstream's installer changes inside Remi's `default_torch_command()`: torch 2.14.0+cu130 / torchvision 0.29.0+cu130, no Nunchaku pair; macOS keeps native wheels. `docker/Dockerfile`, `docker/README.md` and `tests/test_launch_platform.py` follow |
| `modules/initialize.py`, `modules/errors.py` (`a02a0516` torch) | Upstream raised the expected torch in `errors.check_versions()` to 2.14.0; Remi's `initialize.check_versions()` had its own copy of the check, left at 2.13.0 by the merge. It now calls `errors.check_versions()` (Gradio check removed) as upstream does, so a later bump merges by itself; `StartupVersionCheckTests` keeps it equal to the launcher pin |
| `modules/script_callbacks.py` (`f5292b0f` callback) | Upstream's `caller_filename()` (no `inspect.stack()`) and quote style merged; the UI callbacks stay removed (`tests/test_script_callbacks_registration.py`) |
| `modules/shared_options.py`, `modules_forge/presets.py` | `gr.*` converted to `ui.*` (see above); the Moss preset merged |
| `scripts/xyz_grid.py` (`ad607938` xyz, `f74f60c5` cost) | Runtime merged: costs and **Enable the cost system for X/Y/Z Plot** (`xyz_use_cost`, needs restart), `Checkpoint name` through `override_settings`, the new `VAE ("state_dict" only)` axis and `VAE` through the module list, the final loading-parameter reset. Remi's `params()` / actions and `refresh_loading_params_for_xyz_grid` kept; axis values travel by label, so the new axis does not shift saved drafts. The legacy contract lists the new axis as an expected change |
| `README.md`, `README.zh-CN.md` | Remi's READMEs kept; Moss-Image and Moss-Image-Edit added to the model-family list |
| `requirements.txt` (`a02a0516`) | Merged as upstream (no Gradio); installed in the Remi venv, `uv pip check` clean |

## Skipped

| Upstream commit | Why |
| --- | --- |
| `acc2e876` refresh (`modules/ui.py`: `main_entry.refresh_models()` before building the UI) | Covered: Remi refreshes the model and module lists at startup (`remi/server/runtime.py:146`) and again on demand (`:189`); `modules/ui.py` stays deleted |
| `a552980e` lint (`modules/ui_extra_networks_checkpoints_user_metadata.py`: quote style; "Preferred VAE / Text encoder(s)" read `main_entry.module_list`) | No change: the file stays deleted, and Remi's checkpoint metadata editor has no preferred VAE / text encoder field (description, sd version, tags, notes: `remi/networks.py:84`) |

## Validation

- `tools/check.sh` with the Remi venv (Python 3.13, torch 2.14.1, Pillow
  13.0.0, macOS on Apple M4): all 59 steps pass (every pytest suite, the
  contracts, `legacy-ui` and `legacy-ui:fixtures`, i18n, build, bundle budget,
  lint, vitest, Playwright e2e against the mock backend).
- Step B gates: no unresolved files or conflict markers, no Gradio
  (`ruff --select TID251`), `check_line_endings.py` (80 CRLF files,
  `tools/upstream_crlf.txt` unchanged after `--update`),
  `sdapi_contract.py --rev upstream/neo`, `LAST_ABSORBED = c6c2b43d`.
- Real-model smoke test: `tools/smoke_real.py --features --models` on an
  Apple M4 (Anima Turbo v1.1, Qwen-Image VAE, Qwen3 0.6B), report
  [docs/research/smoke/2026-10-27.json](research/smoke/2026-10-27.json):
  all 103 checks pass. txt2img through `/sdapi/v1/txt2img` and
  `/api/v2/jobs` is pixel-identical to the committed baseline (mean 0, max 0);
  the only infotext difference is `Version: neo-2.30 -> neo-2.31`. img2img,
  inpaint, Extras (Lanczos, ESRGAN), the LoRA / LoKr / feature round trips
  (each paste regenerates byte-identical), Refiner, ControlNet lineart
  (LLLite) and ADetailer pass too; the server has no Gradio module loaded.
- Still to check by hand on a GPU: a Moss-Image / Moss-Image-Edit generation
  (no Moss weights here), Skip Layer Guidance together with Spectrum on
  Moss-Image, and an X/Y/Z run over the new VAE axes.
