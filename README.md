<div align="center">

# Forge Neo Remi

**An Anima-first, batteries-included distribution of Stable Diffusion WebUI Forge Neo, with its own fast, Gradio-free web interface.**

[English](README.md) | [简体中文](README.zh-CN.md)

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE)
[![Python 3.13](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Branch: remi](https://img.shields.io/badge/branch-remi-8A2BE2)](https://github.com/beautifulrem/sd-webui-forge-classic/tree/remi)
[![Upstream: Forge Neo](https://img.shields.io/badge/upstream-Forge%20Neo-2ea44f)](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)

</div>

Forge Neo Remi is a downstream distribution of [Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo), itself built on [Stable Diffusion WebUI Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge) and [AUTOMATIC1111 WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui). It keeps Forge Neo's generation engine and model support, adds a curated Anima generation stack with conflict-aware controls, and replaces the Gradio interface with a new web application: a FastAPI server, a persistent job queue and a React single-page app.

> [!IMPORTANT]
> This is an independent downstream branch, not an official release of Forge Neo or the Anima model. Remi favors integrated, tested workflows over compatibility with every third-party extension: extensions written for the old Gradio UI must be ported (see [Migrating from the Gradio UI](#migrating-from-the-gradio-ui)).

**Contents:** [The Remi WebUI](#the-remi-webui) · [Anima integration](#anima-integration) · [Supported models](#supported-model-families) · [Quick start](#quick-start) · [Launch options](#launch-options) · [Where data is stored](#where-data-is-stored) · [Keyboard shortcuts](#keyboard-shortcuts) · [Migrating from the Gradio UI](#migrating-from-the-gradio-ui) · [Security and remote access](#security-and-remote-access) · [Troubleshooting](#troubleshooting) · [Documentation](#documentation)

## Why Remi?

- **A new web interface.** One app with a navigation rail instead of Gradio tabs: a three-pane Generate view, a real image editor, a gallery, a networks browser and a queue; light and dark themes, phone layouts, full keyboard operation, and English / 简体中文 switchable without a reload.
- **One queue for everything.** Every generation is a job in a persistent queue that survives restarts; queue, reorder, prioritize, edit and repeat jobs from the UI or the API.
- **Anima-first inference:** native samplers, guidance methods, spatial conditioning, LoRA controls, Hires safeguards and acceleration paths.
- **28/40/52-block support:** detects original, 2.9B and 3.8B Anima DiT layouts; 28-block LoRAs and Anima ControlLLLite models are remapped onto the expanded models, and Layer Weight, stage-scheduler block weights and FreeFuse's collect block keep following the 28-block layout there. 3.8B needs an external conditioning extension (see [Supported Model Families](#supported-model-families)).
- **Conflict-aware UI and runtime:** mutually exclusive features are locked or resolved before they can form undefined sampling stacks, in the browser and for API jobs alike.
- **One-click safe baseline:** **Anima Remi Presets** provides a conservative starting profile without replacing prompts, LoRA names or model paths.
- **Batteries included:** frequently used extensions ship as built-ins, ported to the new UI.

## The Remi WebUI

The browser app (`webui/`) talks to the server only through a documented API ([`/api/v2`](docs/remi-webui/api.md) plus a WebSocket for progress and live previews); the server never renders HTML. Every page is described below.

**Shell.** A slim navigation rail on the left (a bottom bar and a "More" sheet on phones) leads to the pages; extension pages appear between the built-in pages and Settings. The header shows the page title, the **model bar** (UI preset, checkpoint with architecture and size, VAE / text encoders, "Diffusion in Low Bits"), the ⚡ **quick settings** you pinned, a **Networks** button, the **command palette** (Ctrl/⌘+K: every page, command and shortcut, searchable), a connection indicator and the **Appearance** menu (light / dark / system theme, comfortable or compact density, accent color, language). A thin progress line runs under the header while a job generates.

**Generate (txt2img and img2img).** One view with a txt2img / img2img switch, each with its own form. On wide screens it has three resizable, collapsible panes:

- *Parameters*: the run controls at the top (Generate, Add to queue, Interrupt, Skip, and a menu with generate forever, generate N times and completion notifications), the prompt and negative prompt in a syntax-highlighting editor (attention weights, LoRA tags, schedules, wildcards, bracket checks, tag autocompletion, token counter, styles), then the parameter cards in your configured order: sampler, size (aspect presets, "W x H" paste), guidance, Hires. fix, seed, batch, and every extension panel as a collapsible section; a searchable list holds the scripts, with an "only modified" filter and per-panel reset. A dot next to a changed value resets it to its default. Values you type are remembered per browser.
- *Stage*: the live preview while a job runs (step, batch, ETA, queue position), then the selected result with actions: download, save, copy parameters, reuse seed, use all parameters, send to img2img / inpaint / Extras. In img2img the stage is the **image editor** for the source image: brush, eraser, fill, rectangle and lasso selections, eyedropper, layers for sketch and inpaint masks, undo/redo, zoom and pan, pen pressure, and overlays that show the resize frame and the "only masked" area.
- *History*: this session's results as a virtualized thumbnail strip; a full-screen viewer with zoom, pan, tile preview and gamepad support.

Below 72 rem the history becomes a strip; on phones the stage comes first and the run controls stick to the bottom of the screen. Drop an image anywhere on the prompt to read its generation parameters.

**Extras.** Upscaling, face restoration and the other postprocessing operations on a single image, a batch, a folder or a video, with a before/after comparison slider.

**PNG Info.** Drop, paste or choose an image to see its generation parameters, every metadata entry (text chunks, EXIF, PNG chunks), and send the parameters or the image to another page.

**Gallery.** Every image in your output folders, indexed: search the prompt, file name or model; filter by form, folder, model, date range and stars; justified or masonry layout; select many to star, download as zip, compare two, or delete. The viewer shows the parameters and can reveal the file in your file manager.

**Networks.** One browser for LoRA / LyCORIS, embeddings, checkpoints, VAE and text encoders, as a full page or a drawer next to the prompt (Alt+N): folder tree, architecture filters ("compatible with the current preset"), sort, card and list views, previews (images or videos), metadata and training tags, an editor for each item's user metadata (description, activation text, preferred weight, …), setting a preview from your results, and a refresh button that rescans the folders (it also updates the model bar). Thousands of items scroll smoothly.

**Queue.** The running job with its live preview, the queued jobs in run order (drag or keyboard to reorder, high / normal / low priority, duplicate, edit, cancel), pause / resume, and the searchable history of every job with its outputs, stars and "queue again". Imports tasks from the old Agent Scheduler and Prompt Queue.

**Settings.** All settings in one searchable, virtualized list with categories, "changed" / "unsaved" / "pinned" filters, per-setting reset, export and import (also of an A1111 `config.json`), plus server tools: Reload UI, restart, unload or reload the model, show the weights in memory, hash all checkpoints, system info and licenses.

**Extensions.** Installed extensions (enable, disable, check for updates, update, uninstall), the extension index with search and tag filters, install from a git URL, and configuration backups you can restore.

**Extension pages.** Extensions add their own pages: **JoyCaption** (image captioning, single or folder, with progress and cancel) and **Prompt Workshop** ship with Remi.

**Prompt tools.** Under every prompt: a tag view (chips you can reorder, reweight, disable or delete), quick tag groups, prompt history, favorites in collections, and optional translation through a provider you configure. Nothing is sent anywhere until you choose a provider.

**API.** The same server offers [`/api/v2`](docs/remi-webui/api.md) (what the SPA uses: named parameters, validation, jobs, WebSocket events) and, with `--api`, the A1111-compatible `/sdapi/v1`. Interactive docs are at `/docs`.

## Anima Integration

| Area | Included capabilities |
| --- | --- |
| Sampling | Beta57, Flow Euler, UniPC2, PC3, SPEED (experimental half-res early steps; switch sigma adjustable in Settings), ER SDE, Dynamic Shift |
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
> Remi only covers the 3.8B **diffusion model** (52-block DiT layout, LoRA mapping). Anima 3.8B is conditioned on Qwen3.5 4B: the v1.1 checkpoint bundles a timestep-aware Semantic Connector (v2) that runs at every denoising step and needs the separate `qwen35_4b` text encoder next to the native Qwen3 0.6B one. Remi contains no Qwen3.5 conditioning or connector code; as in upstream Forge Neo, that path requires the [forge-anima-3.8B](https://github.com/GumGum10/forge-anima-3.8B) extension, which Remi does not bundle and has not tested (it must also be ported to the Remi UI). See the [Anima 3.8B model card](https://huggingface.co/lylogummy/Anima-3.8B).

- SD 1.x and advanced SDXL variants
- Krea 2 and Krea 2 Identity Edit
- FLUX Kontext and FLUX.2 Klein
- Qwen-Image, Qwen-Image-Edit and Qwen-Image-2.1
- Ernie-Image, PiD 1.5, Z-Image, Mugen, Lumina Image 2, and Chroma
- Wan 2.2 image/video workflows

Use the upstream [model download guide](https://github.com/Haoming02/sd-webui-forge-classic/wiki/Download-Models) for checkpoint, VAE and text-encoder layouts.

## Quick Start

### Requirements

- [Git](https://git-scm.com/downloads)
- [Python 3.13](https://www.python.org/downloads/) or [uv](https://docs.astral.sh/uv/)
- A supported PyTorch device; an NVIDIA GPU is the most widely tested option (Apple Silicon works through MPS)
- For the web interface: [Node.js](https://nodejs.org/) 20.19+ or 22.12+ (needed once per update to build it; see [The web interface build](#the-web-interface-build)). pnpm is used if installed; otherwise the launcher runs it through Corepack or `npx`, which come with Node.js.
- Optional: [FFmpeg](https://ffmpeg.org/) for video export

### Install and run

```bash
git clone --branch remi https://github.com/beautifulrem/sd-webui-forge-classic.git forge-neo-remi
cd forge-neo-remi
```

**Windows:** run `webui-user.bat`. For the recommended `uv` setup, create the environment first and add `--uv` to `COMMANDLINE_ARGS` in `webui-user.bat`:

```bat
uv venv venv --python 3.13 --seed
```

**Linux and macOS:** run `./webui.sh`. Fresh installations prefer `python3.13`; otherwise `uv` finds or installs Python 3.13 and creates the virtual environment. Existing environments are reused without being deleted or rebuilt. On macOS the launcher installs native `torch` and `torchvision` from PyPI instead of CUDA wheels. Put your own settings in `webui.settings.sh` at the repository root (ignored by Git):

```bash
export PYTHON="/path/to/python3.13"
export COMMANDLINE_ARGS="--uv"
# Optional: override the automatically selected PyTorch installation command
# export TORCH_COMMAND="pip install torch torchvision"
```

The first launch installs the Python packages, builds the web interface and prints `Running on local URL:  http://127.0.0.1:7860/`; open it in a browser (or start with `--autolaunch`). Put checkpoints and companion modules in the matching `models/` subfolders (`Stable-diffusion`, `VAE`, `text_encoder`, `Lora`, …), then pick them in the model bar. Files added while the WebUI runs appear after the refresh button of the Networks browser (Alt+N) on the matching tab.

See the upstream [Unix installation guide](https://github.com/Haoming02/sd-webui-forge-classic/wiki/Unix) for platform packages. Apple Silicon can use PyTorch MPS, but CUDA-only attention and compilation backends are unavailable and generation is generally slower than on a dedicated GPU.

### The web interface build

The interface is a static app built from `webui/` into `webui/dist`. The launcher (`launch.py`, run by `webui.sh` / `webui-user.bat`) builds it automatically when `webui/dist` is missing or older than its sources: after a fresh clone and after every update that touched `webui/`. It runs `pnpm install --frozen-lockfile` and `pnpm build` with the pnpm it finds (installed pnpm, then Corepack, then `npx pnpm@12`), and stamps the build so unchanged sources are not rebuilt.

Without Node.js on the machine:

- An existing build is used as it is, with a warning that it may be out of date.
- With no build at all the server still starts (the APIs work) and the browser shows a page explaining how to build the interface.
- To run Remi on a machine without Node.js, build once on any machine that has it (the output is plain static files, the same for every OS) and copy the folder over:

  ```bash
  cd webui
  pnpm install --frozen-lockfile   # or: npx pnpm@12 install --frozen-lockfile
  pnpm build                       # writes webui/dist
  ```

  Copy `webui/dist` into the same place of the target installation. Rebuild and copy again after each update that changes `webui/`.

`REMI_SKIP_WEBUI_BUILD=1` turns the automatic build off; `--skip-install` and `--skip-prepare-environment` skip it too.

### Recommended first Anima run

1. Choose a supported Anima checkpoint and its text encoder and VAE in the model bar.
2. Open the **Anima Remi Presets** panel in the Generate view.
3. Apply **Official Base/Aesthetic — Safe (Recommended)**.
4. Generate once with optional guidance, spatial controls, Hires and compilation disabled.
5. Add one advanced feature at a time and compare with the same seed.

The safe preset starts from `ER SDE`, `36` steps, `CFG 4.5` and the model's standard shift behavior. It configures inactive modules reproducibly but does not silently enable experimental stacks. For Anima-Turbo checkpoints pick **Official Turbo** (`Euler`, `10` steps, `CFG 1`), or **Turbo + NAG** to keep the negative prompt at CFG 1 through Normalized Attention Guidance, which needs no unconditional model pass. A warning is logged when a Turbo checkpoint runs with Base settings or the reverse.

## Launch Options

Pass options in `COMMANDLINE_ARGS` (`webui-user.bat`, `webui.settings.sh`) or to `python launch.py`. `python launch.py --help` lists all of them.

**Server and access**

| Option | Purpose |
| --- | --- |
| `--port N` | Port to listen on. Default: the first free port from 7860 (7861 with `--nowebui`) |
| `--listen` | Listen on all interfaces (`0.0.0.0`) so other machines can connect |
| `--server-name HOST` | Listen on a specific address or host name |
| `--subpath /sd` | Serve the app under a path prefix, for a reverse proxy (with or without prefix stripping) |
| `--tls-keyfile KEY --tls-certfile CERT` | Serve HTTPS (session cookies become `Secure`) |
| `--auth user:pass[,user2:pass2]` | Require a login for the web interface, `/api/v2`, files and the WebSocket (formerly `--gradio-auth`) |
| `--auth-file PATH` | Read logins in the `--auth` format from a file, one or more per line (formerly `--gradio-auth-path`) |
| `--api` | Also serve the A1111-compatible `/sdapi/v1` API |
| `--api-auth user:pass[,…]` | HTTP Basic login for `/sdapi/v1` (and for extension routes from API clients) |
| `--api-no-auth-on-localhost` | With `--auth` and no `--api-auth`: let API clients on this machine use `/sdapi/v1` and extension API routes without a login (the old WebUI behaviour) |
| `--nowebui` | API only (`/api/v2`, `/sdapi/v1`, no web interface), default port 7861 |
| `--cors-allow-origins URLS`, `--cors-allow-origins-regex RE` | Allow browser front-ends on other origins, and extra host names for a loopback-only server |
| `--enable-insecure-extension-access` | Allow installing and updating extensions from the UI while listening on the network |
| `--hide-ui-dir-config` | Hide folder settings and "open folder" / "show in folder" actions |
| `--freeze-settings`, `--freeze-settings-in-sections`, `--freeze-specific-settings` | Make settings read-only |
| `--autolaunch` | Open the browser at start |
| `--ngrok TOKEN` | Publish through an ngrok tunnel |
| `--timeout-keep-alive N` | HTTP keep-alive timeout (seconds) |
| `--api-server-stop` | Enable `/sdapi/v1/server-kill|restart|stop` |

**Paths and installation**

| Option | Purpose |
| --- | --- |
| `--data-dir PATH` | Base folder for user data: settings, styles, outputs, extensions, `models/` and `remi-data/` (default: the installation folder) |
| `--model-ref PATH` | Replace the local `models` directory with a central model directory |
| `--ckpt-dirs PATH` (repeatable) | Extra checkpoint folders |
| `--forge-ref-a1111-home PATH`, `--forge-ref-comfy-home PATH`, `--forge-ref-comfy-yaml PATH` | Reuse model folders of an AUTOMATIC1111 or ComfyUI installation, or a ComfyUI `extra_model_paths.yaml` |
| `--uv`, `--uv-symlink`, `--uv-local-cache` | Install packages with `uv` |
| `--skip-install`, `--skip-prepare-environment` | Skip package installation (and the web interface build) |
| `--sage` / `--flash` | Install optional attention packages; availability does not guarantee selection |
| `--disable-extra-extensions`, `--disable-all-extensions` | Start without user extensions, or without any |

**Changed options.** The Gradio-only options are gone: replace `--gradio-auth` / `--gradio-auth-path` with `--auth` / `--auth-file`, and drop `--gradio-debug`, `--gradio-allowed-path` and `--no-gradio-queue`. The WebUI refuses to start with an unknown option, so an old login option can never be silently ignored. `--share` is accepted but only logs a warning (use `--listen`, a reverse proxy or `--ngrok`); `--use-textbox-seed` has no effect; `--theme light|dark|system` is the theme of browsers that have not picked one in the Appearance menu (`?__theme=light` in the URL picks one, as before); `--agent-scheduler-sqlite-file` tells the queue importer where an old Agent Scheduler database is. Do not install every attention backend blindly; native PyTorch attention is often the most compatible choice, see the upstream [extra installation guide](https://github.com/Haoming02/sd-webui-forge-classic/wiki/Extra-Installations).

## Where Data Is Stored

Paths are relative to the data folder (`--data-dir`, by default the installation folder).

| What | Where |
| --- | --- |
| Settings | `config.json` (`--ui-settings-file`) |
| Prompt styles | `styles.csv` |
| Generated images | `output/txt2img-images`, `output/img2img-images`, `output/extras-images`, grids and saved images next to them (Settings → Paths for saving) |
| Models | `models/` (or `--model-ref`) |
| User extensions | `extensions/` |
| **Remi data** | `remi-data/`: `jobs.db` (queue and history), `files.db` (file index, gallery stars), `uploads/` (images you upload, pruned after `remi_upload_retention_days`), `outputs/` (results the save settings did not write to the output folders, pruned the same way), `thumbs/` (thumbnail cache), `networks/` (extracted previews and placeholders), `prompt-tools.db` (prompt history, favorites, completion usage), `secret.key` and `revoked-sessions.json` (login sessions) |
| Your own CSS | `user.css` (applied to the web interface when present) |
| Completion sound | `notification.mp3` in the installation folder |
| Per browser | form values (drafts), appearance, language, pinned layout sizes: the browser's local storage. "Reset all parameters" in the Generate menu returns a form to its defaults |

Keep `remi-data/` when you move an installation or a Docker volume; deleting it clears the queue, the history, the gallery stars and the prompt tools data, never your images in `output/`.

## Keyboard Shortcuts

The command palette (Ctrl/⌘+K) lists every command with its shortcut; the image editor shows its own list with `?`.

| Where | Keys |
| --- | --- |
| Everywhere | Ctrl/⌘+K command palette · Alt+N networks drawer |
| Generate | Ctrl/⌘+Enter generate (while running: interrupt and generate again, or only interrupt with the setting "Revert [Ctrl + Enter] to only interrupt the generation") · Alt+Enter skip · Esc interrupt (outside text fields and dialogs) |
| Prompt editor | Ctrl/⌘+↑ / ↓ change the weight of the selection · Alt+← / → move the tag · Tab or Enter accept a completion · Ctrl/⌘+Z / Y undo / redo |
| Image editor | B brush · E eraser · G fill · M rectangle · L lasso · I eyedropper · H hand · [ / ] brush size · Shift (hold) erase · Ctrl/⌘+Z / Y undo / redo · Ctrl/⌘+X clear layer · Ctrl/⌘+I invert mask · R or 0 fit · 1 actual size · + / − zoom · F maximize · Space+drag pan · ? help |
| Lightbox | ← / → previous / next · Home / End · + / − / 0 / 1 zoom · S download · T tile preview · Esc close |
| Gallery | arrows move · Enter open · Space select (Shift extends) · F star · Ctrl/⌘+A select all · Delete remove · Esc clear selection |
| Networks | / search · arrows move · Enter apply · I metadata · E edit · Shift+F10 item menu |
| Queue | Space pick up a job, ↑ / ↓ move it, Space drop, Esc cancel |
| Extras | Ctrl/⌘+Enter process · Esc interrupt |
| Settings | / search · Ctrl/⌘+S apply |

## Migrating from the Gradio UI

Remi's interface is new; the generation engine, models, settings file, styles, output folders and infotext format are the same, so your files and images carry over unchanged.

| In the old UI | In Remi |
| --- | --- |
| txt2img and img2img tabs | **Generate** page, txt2img / img2img switch in the header |
| img2img sub-tabs (img2img, Sketch, Inpaint, Inpaint sketch, Inpaint upload, Batch) | mode picker in the **Source** card; the image editor is on the stage |
| Checkpoint, VAE / text encoder and preset dropdowns at the top | **model bar** in the header |
| Quicksettings | ⚡ quick settings in the header; pin settings from their row in Settings |
| Extra Networks tabs under the prompt | **Networks** drawer (Alt+N) and **Networks** page |
| "Generate forever" context menu, Repeat Generate | the menu next to Generate: generate forever, generate N times |
| Agent Scheduler tab and "Enqueue" | **Queue** page and **Add to queue** next to Generate (Ctrl+Shift+Enter); its queue settings are kept, the layout ones are listed in [its README](extensions-builtin/agent-scheduler/README.md) |
| Prompt Queue tab | **Queue** page |
| Prompt All-in-One | prompt tools bar under every prompt (history and favorites are imported automatically on first use); what each old feature became is in [its README](extensions-builtin/sd-webui-prompt-all-in-one-neo/README.md) |
| TagComplete | built-in autocompletion (its settings are kept, except those listed with the reason in [its README](extensions-builtin/sd-webui-tagcomplete-neo/README.md)) |
| Bracket checker, mobile layout | built in |
| PNG Info tab | **PNG Info** page |
| Settings tab, "Defaults" and "Actions" | **Settings** page; server actions under "Server and models". `ui-config.json` is no longer read: forms start from defaults, the UI preset and your browser's saved values |
| Extensions tab | **Extensions** page |
| JoyCaption and Prompt Workshop tabs | pages of the same name |
| Checkpoint Merger tab | **Checkpoint Merger** page (the same controls; merges and saves run behind generation with progress and Cancel) |
| `/file=` links | files are served by id (`/api/v2/files/{id}`) |
| `/agent-scheduler/v1/*`, `/prompt-queue/*` APIs | `POST /api/v2/jobs` and `/api/v2/queue/*` ([API guide](docs/remi-webui/api.md)); the old routes answer `410 Gone` |
| `/tacapi/v1/*`, `/physton_prompt/*` APIs | `/api/v2/prompt-tools` (the old routes answer `410 Gone`) |

**Bringing your queue over.** Open **Queue → ⋯ → Import from Agent Scheduler or Prompt Queue…**. The importer reads the old data without changing it and can run again safely (tasks imported before are skipped). Agent Scheduler pending tasks are queued (the queue is paused first so you can review them), finished tasks go to the history with their images, bookmarks become stars. Script arguments stored by position are named through the current panels; tasks whose arguments cannot all be named are kept as *not replayable* and listed in the report: open one in its form to check and queue it. The Agent Scheduler database is found at `extension-data/agent-scheduler/task_scheduler.sqlite3` or wherever `--agent-scheduler-sqlite-file` points; Prompt Queue's `queue.json` under `extension-data/sd-webui-prompt-queue/`.

**Third-party extensions.** Extensions that only process images (no UI) usually keep working. Extensions with a Gradio UI (`ui()`, `on_ui_tabs`, JavaScript) must be ported: a script that still declares `ui()` is disabled and a file that imports Gradio is skipped, with the reason in the console and in `GET /api/v2/app` (`not_ported`); the rest of the WebUI keeps working. The [extension guide](docs/remi-webui/extension-guide.md) explains the port. API clients written for `/sdapi/v1` keep working with `--api`.

## Built-in Extensions

Remi ships a curated set of integrated extensions, all ported to the new interface:

- ADetailer Neo, NegPiP, Detail Daemon, Prompt Anchor, Prompt Blacklist, Post Processing Suite, Lock img2img Steps
- JoyCaption and Prompt Workshop (pages), prompt tools (successor of Prompt All-in-One) and tag completion, localization packs
- Forge ControlNet with preprocessors, ControlLLLite and IP-Adapter, MultiDiffusion, Image Stitch, Soft Inpainting, Never OOM, Sparse / Radial attention, compile and PiD
- Remi's Anima guidance, Regional, FreeFuse, Artist Mixer, DAVE, Cross-Attention Boost, LoRA layer weights and stage scheduling, prompt rescale, resolutions, Dynamic Shift, Hires Guard, Spectrum and presets

Agent Scheduler, Prompt Queue and Repeat Generate became the core Queue. Built-in status means these extensions are versioned with Remi and covered by its tests; it does not imply that every external extension is compatible.

## Security and Remote Access

By default the server listens on `127.0.0.1` only.

- **Logins.** `--auth` / `--auth-file` protect everything: the interface, `/api/v2`, files, the WebSocket, `/internal` and the API docs. The browser gets a login page; sessions last 30 days and survive restarts, and logging out revokes them. Scripts can send HTTP Basic credentials instead. Ten failed logins from one client within a minute lock it out for a minute. `/sdapi/v1` uses `--api-auth` when set; without it, the `--auth` login (HTTP Basic for scripts), also on a loopback-only server, since a reverse proxy or tunnel on the same machine connects from loopback too. Local tools that cannot send credentials can be let in with `--api-no-auth-on-localhost` (requests from this machine that came through no proxy).
- **Exposing the server.** Use `--listen` (or `--server-name`) together with `--auth`, ideally behind a TLS reverse proxy or with `--tls-keyfile` / `--tls-certfile`. `--share` is not supported. Installing and updating extensions from the UI is disabled while listening on the network unless `--enable-insecure-extension-access` is given; "Show in folder" in the Gallery only works for a browser on the same machine, and `--hide-ui-dir-config` removes every action that opens or reveals folders on the server.
- **Cross-site requests are refused.** State-changing requests and WebSocket connections from pages on another site are rejected, so a malicious page cannot drive your WebUI. Scripts and API tools send no `Origin` header and are unaffected; browser front-ends on another origin must be allowed with `--cors-allow-origins`.
- **DNS rebinding is refused.** A loopback-only server answers only to `localhost` / loopback host names (or hosts allowed with `--cors-allow-origins`). A local reverse proxy must pass one of those as `Host`, or allow its public origin.
- **Files** are served by opaque id only, uploads are type- and size-checked (`remi_upload_max_mb`), and extension HTML, infotext and metadata are rendered as text or sanitized.
- **Extension routes** (added by extensions to the server) get the same cross-site and host checks; with `--auth` they also require the login or `--api-auth` credentials (local API clients are exempt only with `--api-no-auth-on-localhost`).
- **Startup warnings** name risky combinations: reachable without a login, `--cors-allow-origins=*`, insecure extension access, logins without TLS. See [docs/remi-webui/security.md](docs/remi-webui/security.md) for the threat model and a checklist for running exposed.
- **Outbound requests** happen only for what you use or configure: the extension index on the Extensions page, a translation provider you set up, and model downloads of features that need them on first use (CLIP Modulation, JoyCaption, ControlNet preprocessors, ADetailer).

## Troubleshooting

**"The web interface is not built"** in the browser. Node.js was not found when the launcher ran, the build failed, or it was skipped (`--skip-install`, `--skip-prepare-environment`, `REMI_SKIP_WEBUI_BUILD`); the console says which. An older Node.js gets the warning "Node.js vX is too old to build the web interface" and the build usually fails. Install Node.js 20.19+ / 22.12+ and restart, or build `webui/dist` elsewhere and copy it ([The web interface build](#the-web-interface-build)).

**The build fails** (`pnpm install` or `pnpm build` exits with an error). Run `pnpm install --frozen-lockfile && pnpm build` in `webui/` to see the full message. Behind a proxy, configure it for npm/pnpm (`HTTPS_PROXY`). The server still starts; the API works meanwhile.

**The page looks outdated after an update.** Reload the page; the app also reloads itself once when it notices a new build. If it persists, the build did not run (see above).

**403 "Unknown host refused".** You opened a loopback-only server through another host name (a LAN name, a reverse proxy, a tunnel). Use `--listen` / `--server-name`, or allow the name with `--cors-allow-origins=https://your.host`, or make the proxy send `Host: localhost`.

**403 "Cross-site request refused".** A page on another origin (a separate front-end, a userscript site) sent a write request. Allow that origin with `--cors-allow-origins`. Behind a reverse proxy, forward the public host and port (`X-Forwarded-Host`, `X-Forwarded-Port`, `X-Forwarded-Proto`).

**No live preview or progress; the connection dot in the header is red.** The WebSocket (`/api/v2/ws`) is blocked. A reverse proxy must forward WebSocket upgrades (nginx: `proxy_http_version 1.1; proxy_set_header Upgrade $http_upgrade; proxy_set_header Connection "upgrade";`). Also check that live previews are enabled in Settings.

**Under a reverse proxy path the page is blank.** Start with `--subpath /your/prefix` matching the proxy location.

**The port is taken.** Without `--port` the server tries 7860, 7861, … up to 7879; with `--port` it fails if that port is busy. Look for `Running on local URL:` in the console.

**An extension's panel or page is missing.** Check the console: a script that declares Gradio `ui()` instead of `params()`, or imports Gradio, is disabled with a message (also listed in `GET /api/v2/app` under `not_ported`). Disabled extensions are managed on the Extensions page; changes apply after "Reload UI".

**"Extension access is disabled".** The server listens on the network; start with `--enable-insecure-extension-access` if you trust everyone who can reach it.

**A form keeps odd values.** Form values are saved per browser. Use "Reset all parameters" in the Generate menu, or the reset dot next to a single value.

**Login does not stick.** The session cookie is `Secure` with `--tls-*` or behind HTTPS; open the HTTPS address. Cookies are scoped to `--subpath`.

**Out of memory or slow generation.** These are engine matters as in Forge Neo: try the Never OOM panel, a low-bit dtype in "Diffusion in Low Bits", smaller batches, and native PyTorch attention.

## Documentation

- [Documentation index](docs/remi-webui/README.md)
- [Remi Anima generation and parameter guide](docs/remi-anima-generation-guide.zh-CN.md)
- [API guide (REST, WebSocket, /sdapi)](docs/remi-webui/api.md)
- [Extension developer guide](docs/remi-webui/extension-guide.md)
- [WebUI architecture](docs/remi-webui/architecture.md) and [Anima feature seams](docs/remi-architecture-notes.md)
- [Absorbing Forge Neo updates](docs/remi-webui/upstream-absorption.md)
- [Anima generation optimization research](docs/research/anima-generation-optimizations.md)
- [Forge Neo Wiki](https://github.com/Haoming02/sd-webui-forge-classic/wiki) · [Model downloads](https://github.com/Haoming02/sd-webui-forge-classic/wiki/Download-Models) · [Inference references](https://github.com/Haoming02/sd-webui-forge-classic/wiki/Inference-References)

## Compatibility Notes

- Training features, legacy model families and older WebUI components removed by Forge Neo are not restored by Remi.
- Extensions written for the Gradio UI need a port; processing-only extensions and `/sdapi/v1` clients usually work unchanged.
- CUDA-only packages such as FlashAttention are not available on every platform.
- Experimental guidance methods can alter image character as well as quality; preserve a baseline seed when comparing them.
- Torch compile (whole-model presets and **Anima per-block**) runs the affected steps uncompiled while NAG, NegPiP, Regional, FreeFuse, Artist Mixer, DAVE or Cross-Attention Boost (during their window steps), or Sparse Attention are active, so these features are never silently ignored by a cached graph; expect no compile speed-up for those runs.
- **Anima FlowMatch** is only meaningful for flow-matching models; on other models (e.g. left selected after switching from an Anima Flow sampler) it falls back to the Simple schedule.
- Real checkpoint output remains the final compatibility test. Startup or unit tests do not prove generation quality for every checkpoint or quantization.

## Issues and Contributions

Before opening an issue, reproduce it on the current `remi` branch with third-party extensions disabled. Include:

- Remi commit hash
- Operating system, browser, GPU/backend, Python and PyTorch versions
- Checkpoint family and quantization format
- Launch arguments and relevant generation parameters
- Full console traceback, the browser console for interface problems, or a minimal reproduction

Use the [Remi issue tracker](https://github.com/beautifulrem/sd-webui-forge-classic/issues) for Remi-specific behavior. Upstream-only issues should be reproduced against [Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo) before being reported there.

Contributions should keep features isolated, preserve non-Anima behavior, add regression coverage for shared sampling/model-loading paths, and document GUI/runtime conflict semantics. `tools/check.sh` runs every check (Python and frontend tests, lint, types, build).

## Upstream and Acknowledgements

Remi builds on the work of [AUTOMATIC1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui), [lllyasviel](https://github.com/lllyasviel/stable-diffusion-webui-forge), [Haoming02](https://github.com/Haoming02/sd-webui-forge-classic), [ComfyUI](https://github.com/Comfy-Org/ComfyUI), and the authors of the bundled extensions and algorithms. Their original licenses and notices remain with their respective components.

## License

The main project is licensed under the [GNU Affero General Public License v3.0](LICENSE). Bundled extensions and vendored components may carry additional compatible licenses; consult the `LICENSE` files in their directories.
