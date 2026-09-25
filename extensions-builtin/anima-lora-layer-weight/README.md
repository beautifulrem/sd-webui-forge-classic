# Anima LoRA/LoKR Layer Weight

English | [中文](README_zh.md)

Anima LoRA/LoKR Layer Weight is a Forge/WebUI extension for controlling Anima LoRA and LoKR patches by model block, module type, and Qwen text-encoder layer. It is designed for Anima-style DiT models where different depth ranges tend to affect composition, character features, and style differently.

## Features

- Separate tabs for regular LoRA and LoKR.
- Target all detected Anima adapters, or only specified file names and aliases.
- Block weights for all detected Anima blocks, including 28-block and 40-block models.
- Module weights for `self_attn`, `cross_attn`, `mlp`, `adaln`, and `norm`.
- Optional Qwen text-encoder layer weights.
- Inline prompt syntax for per-LoRA block/module weights.
- Named presets and an **Apply preset to prompt** button.
- Automatic cache invalidation when the active weight signature changes.
- Settings-page UI language switch: `zh` and `en`.

## Installation

1. Copy this folder to your WebUI/Forge `extensions` directory.
2. Restart WebUI, or use Reload UI.
3. Expand **Anima LoRA/LoKR Layer Weight** in txt2img or img2img.

## Weight Syntax

Block ranges:

```text
0-9=0.6,10-18=1.0,19-27=0.75
```

Module weights:

```text
self_attn=1.0
cross_attn=1.0
mlp=0.8
adaln=1.0
norm=1.0
```

Preset lines:

```text
Balanced=blocks=0-9=0.6,10-18=1.0,19-27=0.75;modules=self_attn=1.0,mlp=0.8;qwen=0-5=1.0
```

Inline prompt syntax:

```text
<lora:name:blocks=0-9=0.8,10-18=1.0,19-27=0.7;modules=self_attn=1.0,mlp=0.8>
```

## Practical Layer Meaning

- Blocks `0-9`: early structure, pose, and composition.
- Blocks `10-18`: character identity, body, outfit, and major visual traits.
- Blocks `19-27`: style, light, texture, and finishing detail.

These are practical defaults rather than strict rules. Different training sets may shift the best ranges.

## Language Setting

Open **Settings**, find **Anima LoRA Layer Weight**, and select `zh` or `en`. Reload UI after changing it.

## Notes

- Missing blocks, modules, or Qwen layers are ignored safely.
- Files without text-encoder training naturally skip Qwen weights.
- If inline block/module syntax is used in the prompt, it overrides the corresponding panel block/module settings for that adapter.
