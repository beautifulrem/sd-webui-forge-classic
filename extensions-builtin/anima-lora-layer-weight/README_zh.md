# Anima LoRA/LoKR 分层权重

[English](README.md) | 中文

Anima LoRA/LoKR 分层权重是一个 Forge/WebUI 插件，用于按 Anima block、模块类型和 Qwen 文本编码器层分别控制 LoRA 与 LoKR patch 权重。它适合 Anima 这类 DiT 模型，因为不同深度通常会更偏向构图、人物特征或画风。

## 功能

- 普通 LoRA 和 LoKR 分开控制。
- 可作用于全部检测到的 Anima 适配器，也可只作用于指定文件名或别名。
- 支持对所有已检测 Anima blocks 分层权重，包含 28 层和 40 层模型。
- 支持 `self_attn`、`cross_attn`、`mlp`、`adaln`、`norm` 模块权重。
- 可选 Qwen 文本编码器层权重。
- 支持在提示词内直接写单 LoRA 的 block/module 权重。
- 支持命名预设和 **应用预设到提示词** 按钮。
- 权重签名变化时会自动刷新 LoRA 缓存。
- 设置页提供界面语言切换：`zh` 和 `en`。

## 安装

1. 将插件文件夹复制到 WebUI/Forge 的 `extensions` 目录。
2. 重启 WebUI，或使用 Reload UI。
3. 在 txt2img 或 img2img 展开 **Anima LoRA/LoKR 分层权重**。

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
norm=1.0
```

预设行：

```text
平衡=blocks=0-9=0.6,10-18=1.0,19-27=0.75;modules=self_attn=1.0,mlp=0.8;qwen=0-5=1.0
```

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
