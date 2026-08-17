# Forge Neo Remi README 格式参考

> 调研日期：2026-08-17  
> 范围：Forge Neo 上游及 ComfyUI、AUTOMATIC1111 WebUI、Diffusers、Fooocus 的官方 GitHub README；仅采用项目自身仓库中的一手资料。

## 结论

`forge-neo-remi` 应将 README 重构为两个内容对等、互相链接的文件：

- `README.md`：默认英文版，也是内容更新的主源；
- `README.zh-CN.md`：完整简体中文版，不与英文混排。

两份文件顶部都使用简短语言导航：

```md
[English](README.md) | [简体中文](README.zh-CN.md)
```

README 的职责应限制为项目定位、核心差异、安装和文档入口。Anima 的完整参数、互斥关系、采样器原理和进阶调优继续放在 `docs/`，README 仅给出推荐预设与链接。这样更接近热门项目“先让用户理解并运行，再引导进入详细文档”的写法。

## 一手项目观察

| 项目 | 顶部设计 | 主体结构与可借鉴点 | 不宜照搬的部分 |
|---|---|---|---|
| [Forge Neo 上游 README](https://github.com/Haoming02/sd-webui-forge-classic/blob/neo/README.md) | 居中标题、分支导航、GUI 截图和项目定位 | Features → Commandline → Installation → Attention Functions → Issues & Requests → Thanks；模型支持与命令行信息完整 | 功能清单过长，缺少双语切换、徽章、明确的 License/Contributing 入口；Remi 不应继续在首页追加大段实现细节 |
| [ComfyUI 官方 README](https://github.com/Comfy-Org/ComfyUI/blob/master/README.md) | 居中标题、短标语、官网/社区/Release/下载徽章和大幅截图 | 很早提供 Get Started；按 Desktop、Portable、Manual 区分安装；再列 Examples、Features、Installing、Running、Support 和 QA | 安装与硬件说明非常长；Remi 可把平台细节放到 Wiki 或独立文档 |
| [AUTOMATIC1111 WebUI 官方 README](https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/README.md) | 标题、一句话定位、截图，风格朴素 | Features → Installation and Running → Contributing → Documentation → Credits；详细内容转移到 Wiki | 超长扁平特性列表可读性有限，Remi 应按“核心能力/Anima 增强/内置扩展”分组 |
| [Hugging Face Diffusers 官方 README](https://github.com/huggingface/diffusers/blob/main/README.md) | 居中品牌图；License、Release、下载、行为准则等少量真实徽章 | 先明确定位与三个核心组件，再进入 Installation、Quickstart、Documentation、Contribution、Ecosystem、Credits、Citation | 库级 Quickstart 代码不适合直接套到 GUI 项目，但“最小可运行路径紧邻安装”值得采用 |
| [Fooocus 官方 README](https://github.com/lllyasviel/Fooocus/blob/main/readme.md) | 居中品牌图，紧接明显的安装入口与产品理念 | 用户导向；Download、硬件、Troubleshooting、模型和自定义配置；表格用于能力/硬件比较，复杂内容折叠 | 技术说明和命令行全文仍偏长；Remi 应以文档链接替代大段高级参数 |

### 双语维护参考

上述四个产品仓库的主 README 本身没有中英文双文件切换，不能声称双语格式直接来自它们。可采用同属 ComfyUI 官方组织的[文档仓库英文 README](https://github.com/Comfy-Org/docs/blob/main/README.md)与[中文 README](https://github.com/Comfy-Org/docs/blob/main/readme/zh-CN.md)模式：不同语言独立成文件、顶部互链，并把英文定义为内容主源。该仓库还明确要求译文镜像原文结构，适合 Remi 保持中英文目录和标题一一对应。

## 推荐文件结构

两份 README 应保持相同章节顺序：

1. **Hero 区域**
   - 居中项目名：`Stable Diffusion WebUI Forge Neo — Remi`；
   - 一句话定位：Forge Neo 的 Anima-first、开箱即用发行分支；
   - `English | 简体中文` 语言导航；
   - GUI 截图；
   - 3–5 个可验证徽章。
2. **Overview / 项目简介**
   - 说明它基于 Haoming02 的 Forge Neo；
   - 用两三句话说明 Remi 的目标、受众和与上游的关系；
   - 明确它不是新的基础模型，也不是 ComfyUI 节点运行器。
3. **Highlights / 核心亮点**
   - Anima 28/40-block 与 2.9B 支持；
   - Remi 推荐预设与 GUI 互斥保护；
   - 原生采样器/调度器与质量增强；
   - 预装且 Forge 化的扩展；
   - 仍保留 SDXL、Flux 等 Forge Neo 能力。
4. **Quick Start / 快速开始**
   - 先给最短的 Windows 路径；
   - 给 clone、Python/uv、启动命令；
   - 告诉用户 checkpoint 和组件放置位置；
   - 给第一次启动的预期结果。
5. **Installation / 安装**
   - Windows、Linux、macOS 分开；
   - GPU/后端要求以简表呈现；
   - 复杂依赖、旧显卡与 AMD 细节外链。
6. **Anima Support / Anima 支持**
   - 用表格列出 Base、Aesthetic、Turbo、2.9B/40-block；
   - 每类只给推荐预设入口，不在 README 重复全部参数；
   - 链接中文生成指南，并为英文版准备对应英文指南或清楚标注当前语言。
7. **Built-in Features and Extensions / 内置功能与扩展**
   - 按用途分组：采样与调度、Guidance、LoRA、区域控制、质量/高分辨率、速度/显存、工作流体验；
   - 表格只写“名称、用途、默认状态、限制”；
   - 不按提交历史罗列。
8. **Compatibility / 兼容性**
   - 模型家族、平台和已知限制；
   - 明确实验功能默认关闭；
   - 对互斥功能只给简表，完整规则链接到指南。
9. **Documentation / 文档**
   - Anima 生成指南；
   - 命令行参数；
   - 模型下载/目录；
   - Troubleshooting；
   - API 或上游 Wiki。
10. **Contributing / 贡献**
    - Issue/PR 的最小要求；
    - 要求复现信息、日志、平台和 checkpoint 类型；
    - 链接独立贡献指南；若尚无 `CONTRIBUTING.md`，先使用简短章节，不能链接不存在的文件。
11. **Credits / 致谢与来源**
    - 分别感谢 AUTOMATIC1111、Forge、Forge Neo、ComfyUI 及移植技术作者；
    - 将第三方组件与许可证清单放入独立文档，README 只给入口。
12. **License / 许可证**
    - 明确仓库主体采用 AGPL-3.0；
    - 链接根目录 `LICENSE`；
    - 提醒内置第三方组件可能有各自许可证。

## 徽章建议

只放能落到真实页面、不会制造虚假质量信号的徽章：

- `License: AGPL-3.0` → 根目录 `LICENSE`；
- `Branch: remi` → `beautifulrem/sd-webui-forge-classic/tree/remi`；
- `Upstream: Forge Neo` → `Haoming02/sd-webui-forge-classic/tree/neo`；
- `Python` → README 中实际支持的 Python 版本说明；
- 若仓库以后有稳定 Release 或 CI，再加入 Release/Tests 徽章。

不建议加入没有对应自动化的 `build passing`、主观的 `production ready`、数量过多的依赖徽章，或与 Remi 分支无关的上游下载量。

Diffusers 的[官方 README 顶部](https://github.com/huggingface/diffusers/blob/main/README.md#readme)展示了 License/Release/行为准则徽章如何直接链接到治理文件；ComfyUI 的[官方 README 顶部](https://github.com/Comfy-Org/ComfyUI/blob/master/README.md#comfyui)展示了产品截图和少量社区/发布徽章的组合。

## 写作与同步规则

- 英文是内容主源，但每次面向用户的功能、安装和兼容性变更必须在同一提交同步中文；
- 两个文件保持相同标题层级、表格列和链接目标，不能把中文 README 简化成摘要；
- 第一屏回答三个问题：这是什么、为什么使用、如何开始；
- 用动词和用户结果描述功能，例如 “Prevents incompatible guidance modes from being enabled together”，而不是只列内部模块名；
- 一段只表达一个主题，避免上游 README 中连续数百行 checkbox 的写法；
- 表格用于模型/平台/扩展的并列比较，`<details>` 只折叠可选的高级参数；
- 不在 README 声称未经目标 GPU 固定 seed 验证的画质或性能提升；
- 版本敏感的模型列表、Python/PyTorch 版本和安装命令应附维护日期，或链接到持续维护的独立文档；
- 保留清楚的上游归属与第三方许可入口，避免让读者误以为所有内置扩展均由 Remi 原创或均采用相同许可证。

## 建议的首屏骨架

```md
<h1 align="center">Stable Diffusion WebUI Forge Neo — Remi</h1>

<p align="center">
  An Anima-first Forge Neo distribution with integrated samplers,
  safe presets, conflict-aware controls, and curated built-in extensions.
</p>

<p align="center">
  <a href="README.md">English</a> |
  <a href="README.zh-CN.md">简体中文</a>
</p>

<p align="center">
  <!-- License / branch / upstream badges -->
</p>

<p align="center">
  <img src="html/ui.webp" width="720" alt="Forge Neo Remi WebUI">
</p>

## Overview

<!-- 2–3 sentences: origin, purpose, intended users -->

## Highlights

<!-- 5–7 user-visible differentiators -->

## Quick Start

<!-- shortest supported installation and launch path -->
```

该骨架综合了 Forge Neo 的项目继承说明、ComfyUI/Diffusers 的居中产品首屏、AUTOMATIC1111 的简明入口和 Fooocus 的用户导向安装方式，同时避免复制任何项目的具体文案。
