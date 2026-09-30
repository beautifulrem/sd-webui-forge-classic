<div align="center">

# Forge Neo Remi

**An Anima-first, batteries-included distribution of Stable Diffusion WebUI Forge Neo.**

[English](README.md) | [简体中文](README.zh-CN.md)

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE)
[![Python 3.13](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Branch: remi](https://img.shields.io/badge/branch-remi-8A2BE2)](https://github.com/beautifulrem/sd-webui-forge-classic/tree/remi)
[![Upstream: Forge Neo](https://img.shields.io/badge/upstream-Forge%20Neo-2ea44f)](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)

<img src="html/ui.webp" width="720" alt="Forge Neo WebUI">

</div>

Forge Neo Remi is a downstream distribution of [Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo), itself built on [Stable Diffusion WebUI Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge) and [AUTOMATIC1111 WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui). It keeps the familiar WebUI workflow while integrating a curated Anima generation stack, compatible built-in extensions, conflict-aware controls, and conservative upstream fixes.

> [!IMPORTANT]
> This is an independent downstream branch, not an official release of Forge Neo or the Anima model. Remi intentionally favors integrated, tested workflows over compatibility with every third-party extension.

## Why Remi?

- **Anima-first inference:** native samplers, guidance methods, spatial conditioning, LoRA controls, Hires safeguards, and acceleration paths.
- **28/40/52-block support:** detects original, 2.9B and 3.8B Anima DiT layouts; 28-block LoRAs are remapped onto the expanded models, and Layer Weight, stage-scheduler block weights and FreeFuse's collect block keep following the 28-block layout there. 3.8B needs an external conditioning extension (see [Supported Model Families](#supported-model-families)).
- **Conflict-aware UI and runtime:** mutually exclusive features are locked or resolved before they can form undefined sampling stacks.
- **One-click safe baseline:** **Anima Remi Presets** provides a conservative starting profile without replacing prompts, LoRA names, or model paths.
- **Batteries included:** frequently used Neo-compatible extensions are shipped as built-ins.
- **Selective upstream sync:** useful fixes are adapted without overwriting Remi's Anima attention hooks, lifecycle cleanup, or sampler guards.

## Anima Integration

| Area | Included capabilities |
| --- | --- |
| Sampling | Beta57, Flow Euler, UniPC2, PC3, SPEED (experimental half-res early steps), ER SDE, Dynamic Shift |
| Guidance | NAG, Momentum Guidance, Skimmed CFG, SMC, CFG-Zero*, APG, TCFG, FDG, DCW, CNS, CFG active range (CFG only within a model-time window: start/end are fractions of the model's denoising time, converted to sigma with `percent_to_sigma`, not step fractions; outside it the negative pass is skipped unless another hook reads it), CLIP Modulation guidance (on first use by default it downloads a pinned adapter and CLIP-L; local files also work) |
| Spatial control | Regional Conditioning, FreeFuse multi-LoRA routing, Artist Scheduled Mixer |
| Diversity | DAVE (early-step DC attenuation for same-prompt layout diversity across seeds) |
| Prompt adherence | Cross-Attention Boost (positive-prompt cross-attention gain in the high-noise steps, norm-matched, for weak tags and bindings) |
| LoRA workflow | Layer weights, stage scheduling, 2B LoRAs on 2.9B/3.8B (28/40/52 blocks), prompt rescaling |
| Reliability | Feature conflict resolver, Hires Guard, generation metadata, safe fallbacks |
| Acceleration | Spectrum forecasting and Anima per-block `torch.compile` (after the first recompile, a changed resolution or batch size reuses one dynamic-shape graph; batch 1 (CFG 1, or steps outside the CFG range) and reference frames still get graphs of their own) |
| Low-bit loading | **Diffusion in Low Bits**: `float8-e4m3fn (Anima scaled)` (per-tensor scaled FP8, same VRAM as plain FP8, more precise) and `int8 (Anima ConvRot)` (INT8 GEMMs; CUDA 13 or `--enable-triton-backend`, SM 8.0+) |

Read the [Remi Anima generation and parameter guide](docs/remi-anima-generation-guide.zh-CN.md) before combining experimental methods.

## Supported Model Families

Remi retains Forge Neo's modern-model focus and adds deeper Anima integration.

- Anima and Anima Edit: 2B (28 blocks), 2.9B (40) and 3.8B (52) checkpoints

> [!NOTE]
> Remi only covers the 3.8B **diffusion model** (52-block DiT layout, LoRA mapping). Anima 3.8B is conditioned on Qwen3.5 4B: the v1.1 checkpoint bundles a timestep-aware Semantic Connector (v2) that runs at every denoising step and needs the separate `qwen35_4b` text encoder next to the native Qwen3 0.6B one. Remi contains no Qwen3.5 conditioning or connector code; as in upstream Forge Neo, that path requires the [forge-anima-3.8B](https://github.com/GumGum10/forge-anima-3.8B) extension, which Remi does not bundle and has not tested. See the [Anima 3.8B model card](https://huggingface.co/lylogummy/Anima-3.8B).
- SD 1.x and advanced SDXL variants
- Krea 2 and Krea 2 Identity Edit
- FLUX Kontext and FLUX.2 Klein
- Qwen-Image and Qwen-Image-Edit
- Ernie-Image, PiD 1.5, Z-Image, Mugen, Lumina Image 2, and Chroma
- Wan 2.2 image/video workflows

Use the upstream [model download guide](https://github.com/Haoming02/sd-webui-forge-classic/wiki/Download-Models) for checkpoint, VAE, and text-encoder layouts.

## Quick Start

### Requirements

- [Git](https://git-scm.com/downloads)
- [Python 3.13](https://www.python.org/downloads/) or [uv](https://docs.astral.sh/uv/)
- A supported PyTorch device; an NVIDIA GPU is the most widely tested option
- Optional: [FFmpeg](https://ffmpeg.org/) for video export

### Clone the Remi branch

```bash
git clone --branch remi https://github.com/beautifulrem/sd-webui-forge-classic.git forge-neo-remi
cd forge-neo-remi
```

### Windows

Run:

```bat
webui-user.bat
```

For the recommended `uv` setup, create the environment first and add `--uv` to `COMMANDLINE_ARGS` in `webui-user.bat`:

```bat
uv venv venv --python 3.13 --seed
```

### Linux and macOS

Run:

```bash
./webui.sh
```

See the upstream [Unix installation guide](https://github.com/Haoming02/sd-webui-forge-classic/wiki/Unix) for platform packages and launch configuration. Apple Silicon can use PyTorch MPS, but CUDA-only attention and compilation backends are unavailable and generation is generally slower than on a dedicated GPU.

The first launch installs the required Python packages. Place checkpoints and companion modules in the appropriate `models/` subdirectories, then restart or refresh the model list.

## Recommended First Anima Run

1. Load a supported Anima checkpoint and its required text encoders/VAE.
2. Open **Anima Remi Presets**.
3. Apply **Official Base/Aesthetic — Safe (Recommended)**.
4. Generate once with optional guidance, spatial controls, Hires, and compilation disabled.
5. Add one advanced feature at a time and compare with the same seed.

The safe preset starts from `ER SDE`, `36` steps, `CFG 4.5`, and the model's standard shift behavior. It configures inactive modules reproducibly but does not silently enable experimental stacks.

For Anima-Turbo checkpoints pick **Official Turbo** (`Euler`, `10` steps, `CFG 1`), or **Turbo + NAG** to keep the negative prompt at CFG 1 through Normalized Attention Guidance, which needs no unconditional model pass. A warning is logged when a Turbo checkpoint runs with Base settings or the reverse.

## Built-in Extensions

Remi ships a curated set of integrated extensions, including:

- ADetailer Neo, Agent Scheduler Neo, Repeat Generate 4NEO, NegPiP, and Detail Daemon
- Prompt All-in-One Neo, TagComplete Neo, prompt queue/workshop helpers, and localization support
- Forge ControlNet, preprocessors, IP-Adapter, MultiDiffusion, Image Stitch, and post-processing tools
- Remi's Anima guidance, Regional, FreeFuse, Artist Mixer, LoRA scheduling, Hires Guard, Spectrum, PiD, and compile integrations

Built-in status means these extensions are versioned with Remi and participate in its compatibility checks. It does not imply that every external extension is compatible.

## Common Launch Options

| Option | Purpose |
| --- | --- |
| `--uv` | Use `uv pip` for dependency installation |
| `--model-ref PATH` | Replace the local `models` directory with a central model directory |
| `--forge-ref-a1111-home PATH` | Reuse model directories from an AUTOMATIC1111 installation |
| `--forge-ref-comfy-home PATH` | Reuse model directories from a ComfyUI installation |
| `--forge-ref-comfy-yaml PATH` | Read ComfyUI `extra_model_paths.yaml` |
| `--sage` / `--flash` | Install optional attention packages; availability does not guarantee selection |
| `--agent-scheduler-trust-unsigned-params` | Once, after upgrading: sign Agent Scheduler tasks saved by older versions (see below) |

Do not install every attention backend blindly. Native PyTorch attention is often the most compatible choice. Run `python launch.py --help` and consult the upstream [extra installation guide](https://github.com/Haoming02/sd-webui-forge-classic/wiki/Extra-Installations) for additional flags.

## Security and Remote Access

Remi hardens the routes that bundled extensions add to the WebUI server. This matters most with `--share`, `--listen`, or a reverse proxy:

- **Cross-site writes are refused.** Every route an extension adds refuses state-changing browser requests from another site (`Origin` / `Sec-Fetch-Site`). Scripts and API tools send no `Origin` header and are unaffected. Browser front-ends on another origin must be allowed with `--cors-allow-origins` / `--cors-allow-origins-regex`.
- **DNS rebinding is refused.** Without `--listen`, a non-loopback `--server-name`, `--share` or `--ngrok` the server only listens on loopback, so extension routes only answer to `localhost` / loopback addresses (or a host allowed with `--cors-allow-origins`). A local reverse proxy must pass one of those as `Host`, or allow its public origin with `--cors-allow-origins`.
- **The Gradio login covers extension APIs.** With `--gradio-auth`, the routes of Agent Scheduler, Prompt All-in-One (prompt history, stored API keys), TagComplete, and Prompt Queue require the same login as the UI. Agent Scheduler keeps using HTTP Basic auth when `--api-auth` is set.
- **Exposed servers log in on every extension route.** With `--share`, `--ngrok`, `--listen` or a non-loopback `--server-name`, and `--gradio-auth` / `--gradio-auth-path`, routes that other extensions add also require the UI login. Scripts and API tools (including ones on your LAN) then authenticate with `--api-auth` HTTP Basic credentials (as for `/sdapi`); without `--api-auth` they are refused. With `--nowebui` (API only) there is no UI login: whenever `--api-auth` is set, those routes require its credentials on any bind, loopback included, exactly as `/sdapi` does. Loopback-only UI servers keep those routes open to API tools.
- **Reverse proxies** must forward the public host. If the proxy drops a non-default public port from `Host` (e.g. nginx `$host` on `:8443`), also send `X-Forwarded-Port`; otherwise UI saves are refused as cross-site.
- **Agent Scheduler task params are signed.** Queued tasks store pickled script arguments, so they are signed with a per-install key (`agent_scheduler_signing.key`, next to the task database). Keep this file with the database when moving an install or a Docker volume. `/import` only accepts queues exported from the same install. The task database and key are never served through `/file=`.
- **Outbound requests** made by Agent Scheduler (API callbacks, image URLs) are checked against the address actually connected to: cloud-metadata and link-local targets are refused, and image URLs follow Forge's *Settings → API* request policy.

### Upgrading from an older Remi or Agent Scheduler

Tasks saved before signing existed will not run and fail with a message saying so. If nobody else could have imported tasks into your database, start once with `--agent-scheduler-trust-unsigned-params`, then remove the flag; tasks that failed for being unsigned can be requeued from the History tab.

## Documentation

- [Remi Anima generation and parameter guide](docs/remi-anima-generation-guide.zh-CN.md)
- [Anima generation optimization research](docs/research/anima-generation-optimizations.md)
- [Remi architecture notes](docs/remi-architecture-notes.md): feature seams and deliberate trade-offs
- [Forge Neo Wiki](https://github.com/Haoming02/sd-webui-forge-classic/wiki)
- [Model downloads](https://github.com/Haoming02/sd-webui-forge-classic/wiki/Download-Models)
- [Inference references](https://github.com/Haoming02/sd-webui-forge-classic/wiki/Inference-References)

## Compatibility Notes

- Training features, legacy model families, and older WebUI components removed by Forge Neo are not restored by Remi.
- Third-party extensions may depend on APIs that Neo or Remi intentionally changed.
- CUDA-only packages such as FlashAttention are not available on every platform.
- Experimental guidance methods can alter image character as well as quality; preserve a baseline seed when comparing them.
- Torch compile (whole-model presets and **Anima per-block**) runs the affected steps uncompiled while NAG, NegPiP, Regional, FreeFuse, Artist Mixer, DAVE or Cross-Attention Boost (during their window steps), or Sparse Attention are active, so these features are never silently ignored by a cached graph; expect no compile speed-up for those runs.
- **Anima FlowMatch** is only meaningful for flow-matching models; on other models (e.g. left selected after switching from an Anima Flow sampler) it falls back to the Simple schedule.
- Real checkpoint output remains the final compatibility test. Startup or unit tests do not prove generation quality for every checkpoint or quantization.

## Issues and Contributions

Before opening an issue, reproduce it on the current `remi` branch with third-party extensions disabled. Include:

- Remi commit hash
- Operating system, GPU/backend, Python, and PyTorch versions
- Checkpoint family and quantization format
- Launch arguments and relevant generation parameters
- Full console traceback or a minimal reproduction

Use the [Remi issue tracker](https://github.com/beautifulrem/sd-webui-forge-classic/issues) for Remi-specific behavior. Upstream-only issues should be reproduced against [Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo) before being reported there.

Contributions should keep features isolated, preserve non-Anima behavior, add regression coverage for shared sampling/model-loading paths, and document GUI/runtime conflict semantics.

## Upstream and Acknowledgements

Remi builds on the work of [AUTOMATIC1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui), [lllyasviel](https://github.com/lllyasviel/stable-diffusion-webui-forge), [Haoming02](https://github.com/Haoming02/sd-webui-forge-classic), [ComfyUI](https://github.com/Comfy-Org/ComfyUI), and the authors of the bundled extensions and algorithms. Their original licenses and notices remain with their respective components.

## License

The main project is licensed under the [GNU Affero General Public License v3.0](LICENSE). Bundled extensions and vendored components may carry additional compatible licenses; consult the `LICENSE` files in their directories.
