# Anima LoRA/LoKR 分层权重

[English](README.md) | 中文

Anima LoRA/LoKR 分层权重是一个 Forge/WebUI 插件，用于按 Anima block、模块类型和 Qwen 文本编码器层分别控制 LoRA 与 LoKR patch 权重。它适合 Anima 这类 DiT 模型，因为不同深度通常会更偏向构图、人物特征或画风。

## 功能

- 普通 LoRA 和 LoKR 分开控制。
- 可作用于全部检测到的 Anima 适配器，也可只作用于指定文件名或别名。
- 支持对所有已检测 Anima blocks 分层权重：2B（28 层）、2.9B（40 层）与 3.8B（52 层）模型。2B LoRA 在更深的模型上保持自己的层编号。
- 支持 `self_attn`、`cross_attn`、`mlp`、`adaln` 模块权重（Anima block 的 LayerNorm 没有权重，因此没有 `norm` 模块）。
- 可选 Qwen 文本编码器层权重。
- 支持在提示词内直接写单 LoRA 的 block/module 权重。
- 支持命名预设和 **应用预设到提示词** 按钮。
- 权重签名变化时会自动刷新 LoRA 缓存。
- 设置页提供界面语言切换：`zh` 和 `en`。

## 使用

已内置于 Forge Neo Remi：在 txt2img 或 img2img 展开 **Anima LoRA/LoKR 分层权重**。

## 权重语法

Block 范围：

```text
0-9=0.6,10-18=1.0,19-27=0.75
```

模块权重：

```text
self_attn=1.0
cross_attn=1.0
mlp=0.8
adaln=1.0
```

预设行：

```text
平衡=blocks=0-9=0.6,10-18=1.0,19-27=0.75;modules=self_attn=1.0,mlp=0.8
```

预设只含 blocks 和 modules，Qwen 层权重始终来自面板。

提示词内联语法：

```text
<lora:name:blocks=0-9=0.8,10-18=1.0,19-27=0.7;modules=self_attn=1.0,mlp=0.8>
```

## 常用层含义

- Blocks `0-9`：浅层结构、姿态、构图。
- Blocks `10-18`：人物身份、身体、服装和主要特征。
- Blocks `19-27`：画风、光影、质感和细节收尾。

这些是实用经验，不是硬规则。不同训练集的最佳范围可能不同。

## 语言设置

进入 **设置**，找到 **Anima LoRA Layer Weight / Anima LoRA 分层权重**，选择 `zh` 或 `en`，然后 Reload UI。

## 注意

- 文件中不存在的 block、模块或 Qwen 层会安全跳过。
- 没训练文本编码器的文件会自然跳过 Qwen 权重。
- 如果提示词中使用了内联 block/module 语法，它会覆盖该适配器对应的面板 block/module 设置。
- 某个标签页启用时，它控制的每个适配器（该类型的全部已检测 Anima 适配器，或只限列出的目标）在提示词里的权重都会被替换为 `1.0`：`<lora:name:0.6>` 按 `<lora:name:1.0>` 运行，后面的第二个权重（`<lora:name:0.6:0.3>`）也会被丢弃。强度请改在面板里设置：DiT 用 block 权重（如 `all=0.6`），文本编码器用 Qwen 层权重。不受该标签页控制的适配器保留提示词权重。
