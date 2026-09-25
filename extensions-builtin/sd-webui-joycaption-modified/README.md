# sd-webui-joycaption

Adds a **JoyCaption** tab to [Forge Classic (neo)](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo) (and other A1111-style WebUIs) for local image captioning with [JoyCaption Beta One](https://huggingface.co/fancyfeast/llama-joycaption-beta-one-hf-llava).

JoyCaption is free, open, uncensored, and covers both SFW and NSFW content equally — which is exactly what you want when captioning a dataset for LoRA training on something like [Anima](https://huggingface.co/circlestone-labs/Anima).

The tab sits up top alongside `Extras`, `PNG Info`, `Checkpoint Merger`, etc.

---

## Features

- **Single image** tab — drop an image in, get a caption out.
- **Batch (folder)** tab — point at a folder, get a `.txt` sidecar next to every image. Ready to drop straight into LoRA training (kohya_ss, OneTrainer, etc.).
- **All 12 caption styles** from the official HF Space:
  Descriptive (Formal), Descriptive (Casual), Straightforward, Stable Diffusion Prompt, MidJourney, Danbooru / e621 / Rule34 / Booru-like tag lists, Art Critic Analysis, Product Listing, Social Media Post.
- **All 27 extra options** from the official Space (lighting, camera, SFW/NSFW tag, vulgar slang toggle, "avoid meta phrases" for T2I training, etc.).
- **Caption length** as named bucket (`very short` → `very long`) or specific word count.
- **Custom prompt** override for power users.
- **Sampling controls** — temperature, top-p, top-k, max new tokens, system prompt.
- **Three precision modes**:
  - `bf16` — best quality, ~17 GB VRAM (default, fits your 4090 fine)
  - `int8` — ~10 GB VRAM (needs `bitsandbytes`)
  - `nf4` — ~6 GB VRAM (needs `bitsandbytes`)
- **Unload model** button to free VRAM between captioning and image gen.
- **Auto-download** of the model weights to `models/joycaption/` on first run — zero manual file copying.
- **Send-to-txt2img bridge** — after captioning, you get two buttons under the editable combined-prompt box:
  - **→ Append to txt2img Prompt** — adds the caption to whatever is already in your txt2img positive prompt (with smart comma joining).
  - **↻ Replace txt2img Prompt** — overwrites the txt2img positive prompt outright. Designed to pair with [sd-webui-prompt-anchor](https://github.com/<your-fork>/sd-webui-prompt-anchor): your styles / quality tags live in the anchor box up top and stay put, so REPLACING the (subject-only) main prompt with a fresh caption is exactly what you want each time. Without the anchor extension, it's still a quick way to swap prompts wholesale.

---

## Installation

### Option A — git clone (recommended)

From the root of your Forge install:

```bash
cd extensions
git clone https://github.com/<your-fork>/sd-webui-joycaption.git
```

Then restart Forge.

### Option B — drop in

Just unzip the extension into `<forge>/extensions/sd-webui-joycaption/` so you end up with this layout:

```
extensions/
└── sd-webui-joycaption/
    ├── install.py
    ├── README.md
    ├── scripts/
    │   └── joycaption.py
    └── joycaption_lib/
        ├── __init__.py
        ├── prompts.py
        └── model.py
```

Restart Forge.

### Optional — enable 4-bit / 8-bit

If you want the lighter `int8` / `nf4` precision modes, launch Forge with `--bnb` once so it installs `bitsandbytes`:

```bat
set COMMANDLINE_ARGS=--bnb
```

You can remove the flag after the first launch. (Forge's own README documents this.) If `bitsandbytes` isn't installed, only `bf16` shows up in the precision dropdown — everything else still works.

---

## What gets downloaded, and where

**You do not need to manually download anything.** On the first time you press "Load model" or run a caption, the extension calls `huggingface_hub.snapshot_download()` and pulls the weights from [fancyfeast/llama-joycaption-beta-one-hf-llava](https://huggingface.co/fancyfeast/llama-joycaption-beta-one-hf-llava) into:

```
<forge>/models/joycaption/llama-joycaption-beta-one-hf-llava/
```

About **17 GB** in BF16 safetensors. One-time.

If you'd rather download manually (e.g. you already have it cached elsewhere), the files you need from the repo are:

| File | Required |
|---|---|
| `config.json` | ✅ |
| `model-00001-of-0000N.safetensors` (all shards) | ✅ |
| `model.safetensors.index.json` | ✅ |
| `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json` | ✅ |
| `processor_config.json`, `preprocessor_config.json` | ✅ |
| `chat_template.*` | ✅ |
| `generation_config.json` | ✅ |

Drop the folder at `<forge>/models/joycaption/llama-joycaption-beta-one-hf-llava/` and the extension will use it directly without re-downloading.

---

## Usage for Anima LoRA captioning

1. Restart Forge → click the new **JoyCaption** tab at the top.
2. Pick `bf16` (you have 24 GB — use it), check "Keep model loaded", hit **Load model**. First time it'll spend a few minutes downloading.
3. Go to the **Batch (folder)** subtab.
4. Paste the absolute path to your training images, e.g. `D:\datasets\anima-lora\subject`.
5. Caption type: **Descriptive (Formal)** is the safest default. Length: `long` or `very long`.
6. Recommended extra options for diffusion training:
   - ✅ *"Your response will be used by a text-to-image model, so avoid useless meta phrases…"*
   - ✅ *"Include whether the image is sfw, suggestive, or nsfw."* (if mixing sets)
   - ✅ *"Do NOT mention the image's resolution."*
   - For NSFW sets you may also want: *"Do NOT use polite euphemisms — lean into blunt, casual phrasing."*
   - **Ignore** the artist/character-tag style options — they don't help here per your project notes.
7. Click **Run batch**. You'll get a streaming log; each image gets a `.txt` sidecar next to it.

---

## Troubleshooting

- **"`bitsandbytes` not installed" warning** — only relevant if you actually want 4-bit / 8-bit. Ignore for bf16.
- **First load is slow** — that's the model snapshot download. Watch the Forge console window for progress.
- **OOM on a 4090** — close other GPU apps, or switch to `nf4`. JoyCaption itself is ~17 GB in bf16, fits on a 4090 if you're not also loading SDXL/Anima at the same time. Use the **Unload (free VRAM)** button between captioning and image generation.
- **Captions are weirdly stilted in Casual mode** — known JoyCaption quirk per its author; prefer Descriptive (Formal) or Straightforward for training data.
- **Refusals on NSFW prompts** — re-run, or tweak the system prompt. JoyCaption isn't intentionally censored but Llama 3.1's safety layer occasionally trips.

---

## Credits

- [fpgaminer / JoyCaption](https://github.com/fpgaminer/joycaption) — the model itself.
- [Haoming02 / sd-webui-forge-classic (neo)](https://github.com/Haoming02/sd-webui-forge-classic) — the host WebUI.
- Caption type prompts and extra-options list lifted verbatim from the official [fancyfeast/joy-caption-beta-one](https://huggingface.co/spaces/fancyfeast/joy-caption-beta-one) HF Space, so behavior matches the public demo.
