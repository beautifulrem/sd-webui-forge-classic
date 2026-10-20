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
- infotext 会记录生成时使用的面板设置，粘贴图片参数即可恢复（见“Infotext 与粘贴”）。
- 面板文本为中文或英文，跟随 WebUI 的界面语言（见“旧版设置”）。

## 使用

已内置于 Forge Neo Remi：在 txt2img 或 img2img 展开 **Anima LoRA/LoKR 分层权重**。

## Infotext 与粘贴

任一标签页启用时，图片的 infotext（PNG 参数）会以 `Anima LoRA layer lora ...` 和 `Anima LoRA layer lokr ...` 键记录两个标签页的设置，每个控件一个键：`enabled`（启用）、`targets`（目标）、`blocks enabled`、`blocks`（block 层权重开关与内容）、`modules enabled`、`modules`（模块权重开关与内容）、`qwen enabled`、`qwen`（Qwen 文本编码器层权重开关与内容）。未启用的标签页只记录 `enabled: False`。内容为空或对应开关关闭的文本框不写入，因为本次生成没有用到它。

粘贴这些参数（提示词旁的粘贴按钮，或 PNG 信息页的发送按钮）会恢复两个标签页，中英文界面可互相粘贴，相同提示词和种子即可复现图片。开关开启但未写入的文本框会被清空，生成时的效果与原来的空内容相同。因开关关闭而未写入的文本框，以及只用于改写提示词的预设控件，保持不变。图库下方的发送按钮也会把面板从文生图带到图生图。通过 API 请求传入 infotext 时，同样会恢复这些设置。没有这些键的 infotext（未使用分层权重的生成，或旧版本只写了 `Anima LoRA/LoKR Layer Weight` 摘要行的图片）不会改动面板。

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

## 旧版设置

### 未保留

旧版插件的这个设置没有对应项，原因见下表（`tests/test_remi_extension_settings.py`
检查每个旧设置要么仍是设置，要么列在这张表里）：

| 设置 | 原因 |
|---|---|
| `anima_lora_layer_weight_language` | 面板自己的语言选项（`zh` / `en`，默认 `zh`，Reload UI 后生效）。面板现在和其他面板一样跟随 WebUI 的界面语言：在页面顶部（外观 → 语言）选择 English 或 简体中文，无需重新加载即可切换；设置 *Language*（`localization`）是尚未选择语言的浏览器的默认值。中文文本在 `localizations/zh_CN.json` 中，infotext 可在两种语言之间互相粘贴。 |

## 注意

- 文件中不存在的 block、模块或 Qwen 层会安全跳过。
- 没训练文本编码器的文件会自然跳过 Qwen 权重。
- 如果提示词中使用了内联 block/module 语法，它会覆盖该适配器对应的面板 block/module 设置。
- 某个标签页启用时，它控制的每个适配器（该类型的全部已检测 Anima 适配器，或只限列出的目标）在提示词里的权重都会被替换为 `1.0`：`<lora:name:0.6>` 按 `<lora:name:1.0>` 运行，后面的第二个权重（`<lora:name:0.6:0.3>`）也会被丢弃。强度请改在面板里设置：DiT 用 block 权重（如 `all=0.6`；`all=` 默认值也作用于 blocks 之外的 DiT 模块，如 `final_layer`、`x_embedder`、`t_embedder`），文本编码器用 Qwen 层权重（Qwen 层之外的文本编码器键保持 `1.0`）。不受该标签页控制的适配器保留提示词权重。
