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
- **28/40-block support:** detects original and 2.9B Anima architectures; structurally complete legacy 28-block LoRAs can be remapped when a 40-block model is active.
- **Conflict-aware UI and runtime:** mutually exclusive features are locked or resolved before they can form undefined sampling stacks.
- **One-click safe baseline:** **Anima Remi Presets** provides a conservative starting profile without replacing prompts, LoRA names, or model paths.
- **Batteries included:** frequently used Neo-compatible extensions are shipped as built-ins.
- **Selective upstream sync:** useful fixes are adapted without overwriting Remi's Anima attention hooks, lifecycle cleanup, or sampler guards.

## Anima Integration

| Area | Included capabilities |
| --- | --- |
| Sampling | Beta57, Flow Euler, UniPC2, PC3, ER SDE, Dynamic Shift |
| Guidance | NAG, Momentum Guidance, Skimmed CFG, SMC, FDG, DCW, CNS |
| Spatial control | Regional Conditioning, FreeFuse multi-LoRA routing, Artist Scheduled Mixer |
| LoRA workflow | Layer weights, stage scheduling, 28-to-40-block compatibility, prompt rescaling |
| Reliability | Feature conflict resolver, Hires Guard, generation metadata, safe fallbacks |
| Acceleration | Spectrum forecasting and Anima per-block `torch.compile` for suitable repeated workloads |

Read the [Remi Anima generation and parameter guide](docs/remi-anima-generation-guide.zh-CN.md) before combining experimental methods.

## Supported Model Families

Remi retains Forge Neo's modern-model focus and adds deeper Anima integration.

- Anima and Anima Edit, including 28-block and 40-block/2.9B checkpoints
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

Do not install every attention backend blindly. Native PyTorch attention is often the most compatible choice. Run `python launch.py --help` and consult the upstream [extra installation guide](https://github.com/Haoming02/sd-webui-forge-classic/wiki/Extra-Installations) for additional flags.

## Documentation

- [Remi Anima generation and parameter guide](docs/remi-anima-generation-guide.zh-CN.md)
- [Anima generation optimization research](docs/research/anima-generation-optimizations.md)
- [Forge Neo Wiki](https://github.com/Haoming02/sd-webui-forge-classic/wiki)
- [Model downloads](https://github.com/Haoming02/sd-webui-forge-classic/wiki/Download-Models)
- [Inference references](https://github.com/Haoming02/sd-webui-forge-classic/wiki/Inference-References)

## Compatibility Notes

- Training features, legacy model families, and older WebUI components removed by Forge Neo are not restored by Remi.
- Third-party extensions may depend on APIs that Neo or Remi intentionally changed.
- CUDA-only packages such as FlashAttention are not available on every platform.
- Experimental guidance methods can alter image character as well as quality; preserve a baseline seed when comparing them.
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
