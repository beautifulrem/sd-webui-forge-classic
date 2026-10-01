# Anima LoRA Stage Scheduler

English | [中文](README_zh.md)

Anima LoRA Stage Scheduler is a Forge/WebUI extension for controlling when Anima LoRAs affect the sampling process. It can apply a LoRA only during composition, character-feature, or style stages, and it can use different scheduling settings for the base pass and Hires. fix pass.

## Features

- Stage-based LoRA intervention for Anima models without editing model files.
- Built-in templates for composition, character-feature, and style stages.
- Allow and block modes: make a LoRA active only in a stage, or force it to zero in a stage.
- Shift-aware automatic timing. Higher Shift values move stage windows slightly later.
- Separate base-pass and Hires. fix panels. Hires. fix can inherit base settings or use an independent schedule.
- Optional Hires. fix LoRA disabling for all LoRAs or selected full LoRA tags.
- Per-LoRA template overrides, so different LoRAs can use different stage timing in the same prompt.
- Template management: save, rename, delete, and build custom templates from preset combinations.
- Settings-page UI language switch: `zh` and `en`.

## Installation

Built into Forge Neo Remi: open txt2img or img2img and expand **Anima LoRA Stage Scheduler**.

## Basic Usage

1. Enable **stage scheduling**.
2. Choose a target scope:
   - Auto-detect Anima LoRA: apply to detected Anima LoRAs.
   - All LoRAs in prompt: apply to every LoRA tag in the prompt.
   - Target list only: apply only to names listed in the target field.
3. Select a template such as style-stage, composition-stage, or character-feature-stage.
4. Keep **Auto timing from Shift** enabled for normal Anima/DiT workflows.
5. Generate as usual. The extension adjusts LoRA patch strength at sampling time.

## Timing Controls

- **Start ratio** and **end ratio** are sampling progress values from `0` to `1`: the position in the full noise schedule, so a Hires. fix or img2img pass starts part-way (at about `1 - denoising strength`).
- With auto timing off, the panel's single start/end window replaces the template's stages. For a template that only blocks stages, the window blocks the LoRA; otherwise it applies it.
- **Peak ratio** is the point where the stage curve reaches its strongest value.
- **Stage strength multiplier** multiplies the final LoRA strength inside the active window, with auto timing too.
- **Strength curve** controls how strength changes over time: smooth, hold, triangle, front-heavy, or back-heavy. It shapes the panel's window (manual timing, or a custom template without stages); with auto timing a template's stages bring their own curves, and the infotext then leaves the curve out.
- **Shift** is read from the Forge Shift / Distilled CFG control when available. Automatic templates use it to adjust the timing window.

## Per-LoRA Templates

Use one rule per line:

```text
lora_name=template_name
<lora:name:0.8>=template_name
```

LoRAs without a rule use the panel template.

## Template Management

The template manager can save the current base panel as a template, rename custom templates, delete custom templates, and create combination templates from built-in stage presets. Built-in templates cannot be overwritten, renamed, or deleted.

## Saved Settings

A scheduled image's infotext (PNG parameters) records every panel setting the run used, as `Anima LoRA stage ...` keys: target scope and list, panel weight, template, per-LoRA templates, auto timing, the start/end/peak the run used (computed from Shift under auto timing), stage strength and curve (when the run used it); with Hires. fix, whether its panel was independent (and then its settings too) and the Hires. fix LoRA disabling. Pasting the infotext (the paste button next to the prompt, or PNG Info's send buttons) restores the panels, so the same prompt and seed reproduce the image.

- The prompt keeps its LoRA tags as written (`<lora:x:0.6>`); a non-zero panel weight, which replaced that weight in the run, is recorded as its own key and restored with the panel.
- Choices are recorded as fixed codes (`auto`/`all`/`targets`, `smooth`/`triangle`/..., `none`/`all`/`selected`), so zh and en UIs paste each other's images.
- Custom templates are recorded by name. A template missing on the pasting install leaves the template dropdown unchanged, with a warning in the console.
- Images made before this change carry only a one-line `Anima LoRA Stage Scheduler` summary; pasting them leaves the panel unchanged.

## Language Setting

Open **Settings**, find **Anima LoRA Stage Scheduler**, and set the language to `zh` or `en`. Reload UI after changing the option.

## Notes

- Template keys are kept stable for compatibility, so some built-in template names may remain Chinese internally.
- The extension controls LoRA patch strength during sampling. It does not modify checkpoints or LoRA files.
- While enabled, it switches Forge to on-the-fly (online) LoRA for the run, since a strength that changes every step cannot be merged into the weights. Sampling is slower than with merged LoRAs, as Forge warns; the previous setting is restored after the run.
- For best results, combine this with moderate LoRA weights and avoid stacking many aggressive stage schedules at once.
