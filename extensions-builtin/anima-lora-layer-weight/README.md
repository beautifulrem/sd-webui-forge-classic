# Anima LoRA/LoKR Layer Weight

English | [中文](README_zh.md)

Anima LoRA/LoKR Layer Weight is a Forge/WebUI extension for controlling Anima LoRA and LoKR patches by model block, module type, and Qwen text-encoder layer. It is designed for Anima-style DiT models where different depth ranges tend to affect composition, character features, and style differently.

## Features

- Separate tabs for regular LoRA and LoKR.
- Target all detected Anima adapters, or only specified file names and aliases.
- Block weights for all detected Anima blocks: 2B (28), 2.9B (40) and 3.8B (52) models. A 2B LoRA keeps its own block numbering on the deeper models.
- Module weights for `self_attn`, `cross_attn`, `mlp` and `adaln` (Anima's block LayerNorms have no weights, so there is no `norm` module).
- Optional Qwen text-encoder layer weights.
- Inline prompt syntax for per-LoRA block/module weights.
- Named presets and an **Apply preset to prompt** button.
- Automatic cache invalidation when the active weight signature changes.
- Settings-page UI language switch: `zh` and `en`.

## Usage

Built into Forge Neo Remi: expand **Anima LoRA/LoKR Layer Weight** in txt2img or img2img.

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
```

Preset lines:

```text
Balanced=blocks=0-9=0.6,10-18=1.0,19-27=0.75;modules=self_attn=1.0,mlp=0.8
```

Presets hold blocks and modules; Qwen layer weights always come from the panel.

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
- While a tab is enabled, the prompt weight of every adapter it controls (all detected Anima adapters of that kind, or only the listed targets) is replaced with `1.0`: `<lora:name:0.6>` runs as `<lora:name:1.0>`, and a second weight after it (`<lora:name:0.6:0.3>`) is dropped too. Set the strength in the panel instead: block weights such as `all=0.6` for the DiT, Qwen layer weights for the text encoder. Adapters the tab does not control keep their prompt weight.
