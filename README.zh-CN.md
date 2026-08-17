<div align="center">

# Forge Neo Remi

**一个以 Anima 为重点、开箱即用的 Stable Diffusion WebUI Forge Neo 发行分支。**

[English](README.md) | [简体中文](README.zh-CN.md)

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE)
[![Python 3.13](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Branch: remi](https://img.shields.io/badge/branch-remi-8A2BE2)](https://github.com/beautifulrem/sd-webui-forge-classic/tree/remi)
[![Upstream: Forge Neo](https://img.shields.io/badge/upstream-Forge%20Neo-2ea44f)](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)

<img src="html/ui.webp" width="720" alt="Forge Neo WebUI">

</div>

Forge Neo Remi 是 [Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo) 的下游发行分支，而 Forge Neo 建立在 [Stable Diffusion WebUI Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge) 和 [AUTOMATIC1111 WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) 之上。Remi 保留熟悉的 WebUI 工作流，同时集成经筛选的 Anima 生成技术、Neo 兼容内置扩展、冲突感知控件和保守的上游修复。

> [!IMPORTANT]
> 这是独立的下游分支，不是 Forge Neo 或 Anima 模型的官方发行版。Remi 优先保证集成工作流的可用性和可验证性，不承诺兼容所有第三方扩展。

## 为什么选择 Remi？

- **Anima 优先推理：** 内置采样器、引导方法、空间控制、LoRA 管理、高分辨率保护和加速路径。
- **28/40 层架构兼容：** 自动识别原始 Anima 和 2.9B 架构；当 40 层模型激活时，可对结构完整的旧 28 层 LoRA 进行重映射。
- **GUI 与运行时冲突管理：** 互斥功能会在组成未定义采样链之前被锁定或自动解决。
- **一键安全基线：** **Anima Remi Presets** 提供保守的推荐起点，不会覆盖提示词、LoRA 名称或模型路径。
- **常用功能内置：** 常用的 Neo 适配扩展随 Remi 一起版本化。
- **选择性同步上游：** 吸收有价值的修复，同时保留 Remi 的 Anima attention hook、生命周期清理和采样器保护。

## Anima 集成

| 领域 | 内置能力 |
| --- | --- |
| 采样 | Beta57、Flow Euler、UniPC2、PC3、ER SDE、Dynamic Shift |
| 引导 | NAG、Momentum Guidance、Skimmed CFG、SMC、FDG、DCW、CNS |
| 空间控制 | Regional Conditioning、FreeFuse 多 LoRA 路由、Artist Scheduled Mixer |
| LoRA 工作流 | 分层权重、阶段调度、28 到 40 层兼容、提示词缩放 |
| 稳定性 | 功能冲突解析、Hires Guard、生成元数据、安全回退 |
| 加速 | Spectrum 预测，以及适合重复固定工作负载的 Anima per-block `torch.compile` |

组合实验性方法前，请先阅读 [Remi Anima 生成与参数指南](docs/remi-anima-generation-guide.zh-CN.md)。

## 支持的模型家族

Remi 保留 Forge Neo 对现代模型的支持重点，并对 Anima 提供更深入的集成。

- Anima 和 Anima Edit，包括 28 层与 40 层/2.9B checkpoint
- SD 1.x 和高级 SDXL 变体
- Krea 2 和 Krea 2 Identity Edit
- FLUX Kontext 和 FLUX.2 Klein
- Qwen-Image 和 Qwen-Image-Edit
- Ernie-Image、PiD 1.5、Z-Image、Mugen、Lumina Image 2 和 Chroma
- Wan 2.2 图像/视频工作流

请参考上游的 [模型下载指南](https://github.com/Haoming02/sd-webui-forge-classic/wiki/Download-Models) 放置 checkpoint、VAE 和文本编码器。

## 快速开始

### 环境要求

- [Git](https://git-scm.com/downloads)
- [Python 3.13](https://www.python.org/downloads/) 或 [uv](https://docs.astral.sh/uv/)
- PyTorch 支持的计算设备；NVIDIA GPU 是目前测试最充分的选项
- 可选：导出视频时需要 [FFmpeg](https://ffmpeg.org/)

### 克隆 Remi 分支

```bash
git clone --branch remi https://github.com/beautifulrem/sd-webui-forge-classic.git forge-neo-remi
cd forge-neo-remi
```

### Windows

运行：

```bat
webui-user.bat
```

如果使用推荐的 `uv` 方式，先创建虚拟环境，然后在 `webui-user.bat` 的 `COMMANDLINE_ARGS` 中加入 `--uv`：

```bat
uv venv venv --python 3.13 --seed
```

### Linux 和 macOS

运行：

```bash
./webui.sh
```

平台依赖与启动配置请参考上游 [Unix 安装指南](https://github.com/Haoming02/sd-webui-forge-classic/wiki/Unix)。Apple Silicon 可使用 PyTorch MPS，但无法使用仅支持 CUDA 的 attention 和编译后端，生成速度通常慢于独立 GPU。

首次启动会自动安装 Python 依赖。将 checkpoint 及配套模块放入对应的 `models/` 子目录，然后重启 WebUI 或刷新模型列表。

## 首次 Anima 生成建议

1. 加载支持的 Anima checkpoint 及所需文本编码器/VAE。
2. 打开 **Anima Remi Presets**。
3. 应用 **Official Base/Aesthetic — Safe (Recommended)**。
4. 先关闭可选引导、空间控制、Hires 和编译功能生成一次。
5. 每次只增加一个高级功能，并用相同种子对比。

安全预设以 `ER SDE`、`36` 步、`CFG 4.5` 和模型标准 shift 行为为起点。它会为尚未启用的模块写入可复现默认值，但不会暗中启用实验性功能链。

## 内置扩展

Remi 随分支提供经筛选的内置扩展，包括：

- ADetailer Neo、Agent Scheduler Neo、Repeat Generate 4NEO、NegPiP 和 Detail Daemon
- Prompt All-in-One Neo、TagComplete Neo、提示词队列/工作台辅助和本地化支持
- Forge ControlNet、预处理器、IP-Adapter、MultiDiffusion、Image Stitch 和后处理工具
- Remi 的 Anima Guidance、Regional、FreeFuse、Artist Mixer、LoRA 调度、Hires Guard、Spectrum、PiD 和编译集成

“内置”表示这些扩展随 Remi 分支一起版本化，并参与兼容性检查；不代表所有外部扩展都与 Remi 兼容。

## 常用启动参数

| 参数 | 作用 |
| --- | --- |
| `--uv` | 使用 `uv pip` 安装依赖 |
| `--model-ref PATH` | 用集中模型目录替换本地 `models` 目录 |
| `--forge-ref-a1111-home PATH` | 复用 AUTOMATIC1111 安装中的模型目录 |
| `--forge-ref-comfy-home PATH` | 复用 ComfyUI 安装中的模型目录 |
| `--forge-ref-comfy-yaml PATH` | 读取 ComfyUI `extra_model_paths.yaml` |
| `--sage` / `--flash` | 安装可选 attention 包；安装成功不代表一定被选中 |

不要盲目安装所有 attention 后端。PyTorch 原生 attention 通常是兼容性最好的选择。更多参数可运行 `python launch.py --help`，并参考上游 [额外安装指南](https://github.com/Haoming02/sd-webui-forge-classic/wiki/Extra-Installations)。

## 文档

- [Remi Anima 生成与参数指南](docs/remi-anima-generation-guide.zh-CN.md)
- [Anima 生成优化技术研究](docs/research/anima-generation-optimizations.md)
- [Forge Neo Wiki](https://github.com/Haoming02/sd-webui-forge-classic/wiki)
- [模型下载](https://github.com/Haoming02/sd-webui-forge-classic/wiki/Download-Models)
- [推理参考](https://github.com/Haoming02/sd-webui-forge-classic/wiki/Inference-References)

## 兼容性说明

- Forge Neo 已移除的训练功能、旧模型家族和部分旧 WebUI 组件不会由 Remi 恢复。
- 第三方扩展可能依赖 Neo 或 Remi 已主动修改的 API。
- FlashAttention 等仅支持 CUDA 的依赖无法在所有平台上使用。
- 实验性引导方法不仅可能改变质量，也可能改变画面特征；对比时请保留固定种子的基线结果。
- 真实 checkpoint 生成是最终兼容性测试。成功启动或单元测试通过，不能证明所有 checkpoint 和量化都具有相同生成质量。

## 问题反馈与贡献

提交问题前，请先在最新 `remi` 分支上关闭外部第三方扩展并复现。反馈中请包含：

- Remi commit hash
- 操作系统、GPU/后端、Python 和 PyTorch 版本
- checkpoint 家族与量化格式
- 启动参数和相关生成参数
- 完整控制台 traceback 或最小复现步骤

Remi 特有问题请使用 [Remi issue tracker](https://github.com/beautifulrem/sd-webui-forge-classic/issues)。只存在于上游的问题，应先在 [Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo) 上复现，再向上游提交。

贡献应尽量保持功能隔离，不破坏非 Anima 模型行为，为共享采样/模型加载路径增加回归测试，并说明 GUI 与运行时冲突语义。

## 上游与致谢

Remi 基于 [AUTOMATIC1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui)、[lllyasviel](https://github.com/lllyasviel/stable-diffusion-webui-forge)、[Haoming02](https://github.com/Haoming02/sd-webui-forge-classic)、[ComfyUI](https://github.com/Comfy-Org/ComfyUI) 以及所有内置扩展/算法作者的工作构建。各组件保留其原始许可证和声明。

## 许可证

主项目采用 [GNU Affero General Public License v3.0](LICENSE)。内置扩展和引入的组件可能附带其他兼容许可证，请查看对应目录中的 `LICENSE` 文件。
