<div align="center">

# Forge Neo Remi

**一个以 Anima 为重点、开箱即用的 Stable Diffusion WebUI Forge Neo 发行分支，配备独立、快速、不依赖 Gradio 的网页界面。**

[English](README.md) | [简体中文](README.zh-CN.md)

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE)
[![Python 3.13](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Branch: remi](https://img.shields.io/badge/branch-remi-8A2BE2)](https://github.com/beautifulrem/sd-webui-forge-classic/tree/remi)
[![Upstream: Forge Neo](https://img.shields.io/badge/upstream-Forge%20Neo-2ea44f)](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)

</div>

Forge Neo Remi 是 [Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo) 的下游发行分支，而 Forge Neo 建立在 [Stable Diffusion WebUI Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge) 和 [AUTOMATIC1111 WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) 之上。Remi 保留 Forge Neo 的生成引擎与模型支持，集成经筛选的 Anima 生成技术和冲突感知控件，并用全新的网页应用取代了 Gradio 界面：FastAPI 服务器、持久化任务队列和 React 单页应用。

> [!IMPORTANT]
> 这是独立的下游分支，不是 Forge Neo 或 Anima 模型的官方发行版。Remi 优先保证集成工作流的可用性和可验证性，不承诺兼容所有第三方扩展：为旧 Gradio 界面编写的扩展需要移植（见[从 Gradio 界面迁移](#从-gradio-界面迁移)）。

**目录：** [Remi WebUI 界面](#remi-webui-界面) · [Anima 集成](#anima-集成) · [支持的模型](#支持的模型家族) · [快速开始](#快速开始) · [启动参数](#启动参数) · [数据存储位置](#数据存储位置) · [键盘快捷键](#键盘快捷键) · [从 Gradio 界面迁移](#从-gradio-界面迁移) · [安全与远程访问](#安全与远程访问) · [故障排查](#故障排查) · [文档](#文档)

## 为什么选择 Remi

- **全新的网页界面。** 用侧边导航栏取代 Gradio 标签页：三栏式生成页、真正的图像编辑器、图库、模型网络浏览器和任务队列；支持浅色与深色主题、手机布局、完整的键盘操作，English / 简体中文无需刷新即可切换。
- **所有生成共用一个队列。** 每次生成都是持久化队列中的一个任务，重启后仍在；可在界面或 API 中排队、调整顺序、设置优先级、编辑和重复任务。
- **Anima 优先推理：** 内置采样器、引导方法、空间控制、LoRA 管理、高分辨率保护和加速路径。
- **28/40/52 层架构兼容：** 自动识别原始 Anima、2.9B 和 3.8B 的 DiT 结构；28 层 LoRA 和 Anima ControlLLLite 模型会重映射到扩展模型上，分层权重、阶段调度器的 block 权重和 FreeFuse 的采集层在扩展模型上仍按 28 层布局解释。3.8B 还需要外部条件编码扩展（见[支持的模型家族](#支持的模型家族)）。
- **界面与运行时冲突管理：** 互斥功能会在组成未定义采样链之前被锁定或自动解决，浏览器中和 API 任务都一样。
- **一键安全基线：** **Anima Remi 预设** 提供保守的推荐起点，不会覆盖提示词、LoRA 名称或模型路径。
- **常用功能内置：** 常用扩展作为内置扩展随 Remi 发布，并已移植到新界面。

## Remi WebUI 界面

浏览器应用（`webui/`）只通过有文档的 API 与服务器通信（[`/api/v2`](docs/remi-webui/api.md)，外加一个用于进度和实时预览的 WebSocket）；服务器从不渲染 HTML。下面逐页说明。

**整体框架。** 左侧是窄导航栏（手机上为底部栏和“更多”面板），通往各个页面；扩展页面排在内置页面与“设置”之间。顶栏显示页面标题、**模型栏**（界面预设、带架构和大小信息的 checkpoint、VAE / 文本编码器、“低精度扩散”）、你固定的 ⚡ **快捷设置**、**模型网络**按钮、**命令面板**（Ctrl/⌘+K：可搜索所有页面、命令和快捷键）、连接状态指示和**外观**菜单（浅色 / 深色 / 跟随系统、宽松或紧凑密度、强调色、语言）。生成任务运行时，顶栏下方有一条进度线。

**生成（文生图与图生图）。** 同一个视图，顶栏可在文生图 / 图生图之间切换，两者各有独立的表单。宽屏下分为三栏，可调整宽度、可折叠：

- *参数*：顶部是运行控件（生成、加入队列、中止、跳过，以及包含无限生成、连续生成多次和完成通知的菜单）；下面是正向和反向提示词，使用带语法高亮的编辑器（注意力权重、LoRA 标签、提示词调度、通配符、括号检查、标签补全、token 计数、样式）；再往下按你配置的顺序排列参数卡片：采样器、尺寸（宽高比预设、粘贴“W x H”）、引导、高分辨率修复、种子、批次，以及每个扩展面板（可折叠的分区）。可搜索的脚本列表支持“仅显示已修改”筛选和逐面板重置。值旁边的小圆点表示已修改，点击即可恢复默认值。你输入的值会按浏览器保存。
- *舞台*：任务运行时显示实时预览（步数、批次、剩余时间、队列位置），之后显示所选结果及操作：下载、保存、复制参数、复用种子、使用全部参数、发送到图生图 / 局部重绘 / 后期处理。在图生图中，舞台就是源图的**图像编辑器**：画笔、橡皮擦、填充、矩形和套索选区、取色器、涂鸦与重绘蒙版图层、撤销/重做、缩放与平移、压感，以及显示缩放框和“仅蒙版区域”的叠加层。
- *历史*：本次会话的结果，以虚拟化缩略图条显示；全屏查看器支持缩放、平移、平铺预览和手柄操作。

宽度小于 72 rem 时历史栏变为缩略图条；在手机上舞台排在最前，运行控件固定在屏幕底部。把图片拖到提示词上即可读取其生成参数。

**后期处理。** 对单张图片、一批图片、文件夹或视频进行放大、面部修复和其他后处理操作，并提供前后对比滑块。

**图片信息。** 拖入、粘贴或选择图片，查看其生成参数和所有元数据（文本块、EXIF、PNG 块），并把参数或图片发送到其他页面。

**图库。** 为输出文件夹中的所有图片建立索引：搜索提示词、文件名或模型；按表单、文件夹、模型、日期范围和星标筛选；两端对齐或瀑布流布局；多选后可加星标、打包下载、两张对比或删除。查看器显示生成参数，并可在文件管理器中定位文件。

**模型网络。** 一个浏览器统一管理 LoRA / LyCORIS、嵌入、checkpoint、VAE 和文本编码器，可作为整页或提示词旁的抽屉打开（Alt+N）：文件夹树、架构筛选（“与当前预设兼容”）、排序、卡片和列表视图、预览（图片或视频）、元数据与训练标签、条目用户元数据编辑器（描述、激活文本、推荐权重等）、从生成结果设置预览图，以及重新扫描文件夹的刷新按钮（同时更新模型栏）。数千个条目也能流畅滚动。

**队列。** 正在运行的任务及其实时预览，按运行顺序排列的待运行任务（拖动或用键盘调整顺序，高 / 普通 / 低优先级，复制、编辑、取消），暂停 / 继续，以及所有任务的可搜索历史（含输出、星标和“再次排队”）。可导入旧 Agent Scheduler 和 Prompt Queue 的任务。

**设置。** 所有设置集中在一个可搜索的虚拟化列表中，带分类、“已修改” / “未保存” / “已固定”筛选、逐项重置、导出与导入（也支持 A1111 的 `config.json`），以及服务器工具：重新加载界面、重启、卸载或重新加载模型、查看内存中的权重、为所有 checkpoint 计算哈希、系统信息和许可证。

**扩展。** 已安装扩展（启用、禁用、检查更新、更新、卸载）、可搜索并按标签筛选的扩展索引、从 git 网址安装，以及可恢复的配置备份。

**扩展页面。** 扩展可以添加自己的页面：Remi 自带 **JoyCaption**（图片描述，单张或整个文件夹，带进度与取消）和 **Prompt Workshop**。

**提示词工具。** 每个提示词下方都有：标签视图（可重新排序、调整权重、禁用或删除的标签块）、快捷标签组、提示词历史、按收藏夹分组的收藏、通过你配置的服务进行的可选翻译，以及在使用 OpenAI 兼容服务时的“AI 撰写提示词”（由大语言模型根据任意语言的描述撰写或修改提示词）。在你选择翻译服务之前，不会向任何地方发送内容。

**API。** 同一个服务器提供 [`/api/v2`](docs/remi-webui/api.md)（界面所用的 API：命名参数、校验、任务、WebSocket 事件），加上 `--api` 时还提供兼容 A1111 的 `/sdapi/v1`。交互式文档位于 `/docs`。

## Anima 集成

| 领域 | 内置能力 |
| --- | --- |
| 采样 | Beta57、Flow Euler、UniPC2、PC3、SPEED（实验：前段半分辨率；切换 sigma 可在设置中调整）、ER SDE、Dynamic Shift |
| 引导 | NAG、Momentum Guidance、Skimmed CFG、SMC、CFG-Zero*、APG、TCFG、FDG、DCW、CNS、CFG 生效区间（只在模型时间窗口内执行 CFG：起止值是模型去噪时间的比例，经 `percent_to_sigma` 换算为 sigma，不是步数比例；窗口外若无其他 hook 读取，则跳过负向推理）、CLIP Modulation guidance（默认首次使用时下载固定版本的 adapter 与 CLIP-L，也可使用本地文件） |
| 空间控制 | Regional Conditioning、FreeFuse 多 LoRA 路由、Artist Scheduled Mixer |
| 多样性 | DAVE（前段 DC 衰减，提升同一提示词不同 seed 的构图多样性） |
| 提示词遵循 | Cross-Attention Boost（高噪声前段放大正向提示词的 cross-attention 并做范数匹配，改善弱标签与多主体绑定） |
| LoRA 工作流 | 分层权重、阶段调度、2B LoRA 用于 2.9B/3.8B（28/40/52 层）、提示词缩放 |
| 稳定性 | 功能冲突解析、Hires Guard、生成元数据、安全回退 |
| 加速 | Spectrum 预测，以及 Anima per-block `torch.compile`（首次重新编译后，改变的分辨率或 batch size 复用同一个动态形状图；batch 1（CFG 1，或 CFG 区间外的步）和参考帧仍各自编译独立的图） |
| 低比特加载 | **低精度扩散**（Diffusion in Low Bits）中的 `float8-e4m3fn (Anima scaled)`（按张量缩放的 FP8，显存与普通 FP8 相同、精度更高）和 `int8 (Anima ConvRot)`（INT8 矩阵乘；需要 CUDA 13 或 `--enable-triton-backend`，SM 8.0+） |

组合实验性方法前，请先阅读 [Remi Anima 生成与参数指南](docs/remi-anima-generation-guide.zh-CN.md)。

## 支持的模型家族

Remi 保留 Forge Neo 对现代模型的支持重点，并对 Anima 提供更深入的集成。

- Anima 和 Anima Edit：2B（28 层）、2.9B（40 层）与 3.8B（52 层）checkpoint

> [!NOTE]
> Remi 只覆盖 3.8B 的**扩散模型**部分（52 层 DiT 结构与 LoRA 映射）。Anima 3.8B 以 Qwen3.5 4B 为条件：v1.1 checkpoint 内置一个随时间步变化、每个去噪步都参与计算的 Semantic Connector（v2），并需要在原生 Qwen3 0.6B 之外另备 `qwen35_4b` 文本编码器。Remi 没有 Qwen3.5 条件编码或 connector 代码；与上游 Forge Neo 相同，这条路径需要 [forge-anima-3.8B](https://github.com/GumGum10/forge-anima-3.8B) 扩展，Remi 未内置也未测试它（它也需要先移植到 Remi 界面）。详见 [Anima 3.8B 模型卡](https://huggingface.co/lylogummy/Anima-3.8B)。

- SD 1.x 和高级 SDXL 变体
- Krea 2 和 Krea 2 Identity Edit
- FLUX Kontext 和 FLUX.2 Klein
- Qwen-Image、Qwen-Image-Edit 和 Qwen-Image-2.1
- Moss-Image 和 Moss-Image-Edit
- Ernie-Image、PiD 1.5、Z-Image、Mugen、Lumina Image 2 和 Chroma
- Wan 2.2 图像/视频工作流

checkpoint、VAE 和文本编码器的放置方式请参考上游[模型下载指南](https://github.com/Haoming02/sd-webui-forge-classic/wiki/Download-Models)。

## 快速开始

### 环境要求

- [Git](https://git-scm.com/downloads)
- [Python 3.13](https://www.python.org/downloads/) 或 [uv](https://docs.astral.sh/uv/)
- 受支持的 PyTorch 设备；NVIDIA GPU 是测试最充分的选择（Apple Silicon 可通过 MPS 运行）
- 网页界面需要：[Node.js](https://nodejs.org/) 20.19+ 或 22.12+（每次更新后构建一次，见[网页界面的构建](#网页界面的构建)）。已安装 pnpm 时使用 pnpm；否则启动器通过 Node.js 自带的 Corepack 或 `npx` 运行它。
- 可选：用于视频导出的 [FFmpeg](https://ffmpeg.org/)

### 安装与运行

```bash
git clone --branch remi https://github.com/beautifulrem/sd-webui-forge-classic.git forge-neo-remi
cd forge-neo-remi
```

**Windows：** 运行 `webui-user.bat`。推荐使用 `uv`：先创建环境，再在 `webui-user.bat` 的 `COMMANDLINE_ARGS` 中添加 `--uv`：

```bat
uv venv venv --python 3.13 --seed
```

**Linux 和 macOS：** 运行 `./webui.sh`。新安装会优先使用 `python3.13`；若不可用，则由 `uv` 查找或安装 Python 3.13 并创建虚拟环境。已有环境会直接复用，不会删除或重建。在 macOS 上，启动器从 PyPI 安装原生 `torch` 和 `torchvision`，不使用 CUDA wheel。个人设置写在仓库根目录的 `webui.settings.sh` 中（Git 会忽略该文件）：

```bash
export PYTHON="/path/to/python3.13"
export COMMANDLINE_ARGS="--uv"
# 可选：覆盖自动选择的 PyTorch 安装命令
# export TORCH_COMMAND="pip install torch torchvision"
```

首次启动会安装 Python 依赖、构建网页界面，并输出 `Running on local URL:  http://127.0.0.1:7860/`；在浏览器中打开即可（或使用 `--autolaunch` 自动打开）。把 checkpoint 和配套模块放进 `models/` 中对应的子目录（`Stable-diffusion`、`VAE`、`text_encoder`、`Lora` 等），然后在模型栏中选择。WebUI 运行期间新增的文件，在模型网络浏览器（Alt+N）对应标签页中点击刷新按钮后即可出现。

平台依赖请参考上游 [Unix 安装说明](https://github.com/Haoming02/sd-webui-forge-classic/wiki/Unix)。Apple Silicon 可以使用 PyTorch MPS，但 CUDA 专用 attention 和编译后端不可用，生成速度通常也低于独立显卡。

### 网页界面的构建

界面是从 `webui/` 构建到 `webui/dist` 的静态应用。启动器（`launch.py`，由 `webui.sh` / `webui-user.bat` 调用）会在 `webui/dist` 缺失或比源码旧时自动构建：即首次克隆后，以及每次改动了 `webui/` 的更新之后。它使用找到的 pnpm（已安装的 pnpm，其次 Corepack，最后 `npx pnpm@12`）执行 `pnpm install --frozen-lockfile` 和 `pnpm build`，并为构建结果打上标记，源码未变时不会重复构建。

机器上没有 Node.js 时：

- 已有的构建会照常使用，并提示它可能已过期。
- 完全没有构建时服务器仍会启动（API 可用），浏览器会显示一个说明如何构建界面的页面。
- 要在没有 Node.js 的机器上运行 Remi，可在任意一台装有 Node.js 的机器上构建一次（产物是纯静态文件，与操作系统无关），再把目录复制过去：

  ```bash
  cd webui
  pnpm install --frozen-lockfile   # 或：npx pnpm@12 install --frozen-lockfile
  pnpm build                       # 生成 webui/dist
  ```

  把 `webui/dist` 复制到目标安装的相同位置。每次更新改动了 `webui/` 后，都需要重新构建并复制。

设置环境变量 `REMI_SKIP_WEBUI_BUILD=1` 可关闭自动构建；`--skip-install` 和 `--skip-prepare-environment` 也会跳过构建。

### Docker（NVIDIA GPU）

`docker/Dockerfile` 用当前检出的代码构建镜像（构建上下文是仓库根目录）：网页界面在 Node.js 阶段构建，服务器镜像包含 Python 3.13、启动器使用的 PyTorch 版本和 `requirements.txt`，不含 Gradio。全部用户数据放在同一个卷 `/home/remi/data` 中。详见 [docker/README.md](docker/README.md)。

```bash
docker build -f docker/Dockerfile -t remi-webui .
docker run --gpus all -p 7860:7860 -v remi-data:/home/remi/data remi-webui
```

### 首次 Anima 生成建议

1. 在模型栏中选择受支持的 Anima checkpoint 及其文本编码器和 VAE。
2. 在生成页中打开 **Anima Remi 预设** 面板。
3. 应用 **Official Base/Aesthetic — Safe (Recommended)**。
4. 先关闭可选引导、空间控制、高分辨率修复和编译，生成一次。
5. 每次只增加一种高级功能，并使用相同 seed 对比。

安全预设从 `ER SDE`、`36` 步、`CFG 4.5` 和模型标准 shift 行为开始。它会以可复现的方式配置未启用模块，但不会静默开启实验性组合。Anima-Turbo checkpoint 请选择 **Official Turbo**（`Euler`、`10` 步、`CFG 1`），或选择 **Turbo + NAG**，在 CFG 1 下通过 Normalized Attention Guidance 保留反向提示词，无需额外的无条件推理。Turbo checkpoint 使用 Base 设置（或反之）时，会在日志中给出警告。

## 启动参数

把参数写在 `COMMANDLINE_ARGS`（`webui-user.bat`、`webui.settings.sh`）中，或直接传给 `python launch.py`。`python launch.py --help` 会列出全部参数，包括内置和已安装扩展添加的参数（例如 `--agent-scheduler-sqlite-file`、`--controlnet-loglevel`、`--lora-dir`）。

**服务器与访问**

| 参数 | 用途 |
| --- | --- |
| `--port N` | 监听端口。默认：从 7860 起第一个空闲端口（`--nowebui` 时为 7861） |
| `--listen` | 监听所有网卡（`0.0.0.0`），允许其他机器连接 |
| `--server-name HOST` | 监听指定地址或主机名 |
| `--subpath /sd` | 在路径前缀下提供服务，用于反向代理（代理是否去掉前缀均可） |
| `--forwarded-allow-ips 127.0.0.1` | 信任来自这些反向代理（地址或网段，逗号分隔）的 `X-Forwarded-For` / `X-Forwarded-Proto`，从而按真实客户端统计登录失败次数。不设置时忽略这些请求头，代理后的所有客户端共用一个计数 |
| `--tls-keyfile KEY --tls-certfile CERT` | 提供 HTTPS（会话 cookie 变为 `Secure`） |
| `--auth user:pass[,user2:pass2]` | 网页界面、`/api/v2`、文件和 WebSocket 需要登录（原 `--gradio-auth`） |
| `--auth-file PATH` | 从文件读取 `--auth` 格式的登录信息，每行一个或多个（原 `--gradio-auth-path`） |
| `--api` | 同时提供兼容 A1111 的 `/sdapi/v1` API |
| `--api-auth user:pass[,…]` | `/sdapi/v1` 的 HTTP Basic 登录（API 客户端访问扩展路由时也使用） |
| `--api-no-auth-on-localhost` | 设置了 `--auth` 而没有 `--api-auth` 时：本机的 API 客户端无需登录即可使用 `/sdapi/v1` 和扩展 API 路由（旧 WebUI 的行为） |
| `--nowebui` | 仅 API（`/api/v2`、`/sdapi/v1`，无网页界面），默认端口 7861 |
| `--cors-allow-origins URLS`、`--cors-allow-origins-regex RE` | 允许其他来源的浏览器前端，以及仅监听本机时额外允许的主机名 |
| `--enable-insecure-extension-access` | 在网络监听时仍允许从界面安装和更新扩展 |
| `--hide-ui-dir-config` | 隐藏文件夹设置以及“打开文件夹” / “在文件夹中显示”操作 |
| `--freeze-settings`、`--freeze-settings-in-sections`、`--freeze-specific-settings` | 设置只读 |
| `--autolaunch` | 启动时打开浏览器 |
| `--ngrok TOKEN` | 通过 ngrok 隧道发布 |
| `--timeout-keep-alive N` | HTTP keep-alive 超时（秒） |
| `--api-server-stop` | 启用 `/sdapi/v1/server-kill|restart|stop` |

**路径与安装**

| 参数 | 用途 |
| --- | --- |
| `--data-dir PATH` | 用户数据的根目录：设置、样式、输出、扩展、`models/` 和 `remi-data/`（默认：安装目录） |
| `--model-ref PATH` | 用集中的模型目录替代本地 `models` 目录 |
| `--ckpt-dirs PATH`（可重复） | 额外的 checkpoint 目录 |
| `--forge-ref-a1111-home PATH`、`--forge-ref-comfy-home PATH`、`--forge-ref-comfy-yaml PATH` | 复用 AUTOMATIC1111 或 ComfyUI 安装的模型目录，或读取 ComfyUI 的 `extra_model_paths.yaml` |
| `--uv`、`--uv-symlink`、`--uv-local-cache` | 使用 `uv` 安装依赖 |
| `--skip-install`、`--skip-prepare-environment` | 跳过依赖安装（以及网页界面构建） |
| `--sage` / `--flash` | 安装可选 attention 依赖；安装成功不代表一定会被选用 |
| `--disable-extra-extensions`、`--disable-all-extensions` | 不加载用户扩展，或不加载任何扩展 |

**已变更的参数。** 仅用于 Gradio 的参数已移除：请把 `--gradio-auth` / `--gradio-auth-path` 改为 `--auth` / `--auth-file`，并删除 `--gradio-debug`、`--gradio-allowed-path` 和 `--no-gradio-queue`。遇到未知参数时 WebUI 会拒绝启动，因此旧的登录参数不会被悄悄忽略。`--share` 仍被接受，但只会输出警告（请改用 `--listen`、反向代理或 `--ngrok`）；`--use-textbox-seed` 不再起作用；`--theme light|dark|system` 是尚未在外观菜单中选择主题的浏览器所用的主题（与以前一样，URL 中的 `?__theme=light` 会选定主题）；`--agent-scheduler-sqlite-file` 告诉队列导入器旧 Agent Scheduler 数据库的位置。不要盲目安装所有 attention 后端；原生 PyTorch attention 往往兼容性最好，详见上游[额外安装说明](https://github.com/Haoming02/sd-webui-forge-classic/wiki/Extra-Installations)。

## 数据存储位置

以下路径相对于数据目录（`--data-dir`，默认为安装目录）。

| 内容 | 位置 |
| --- | --- |
| 设置 | `config.json`（`--ui-settings-file`） |
| 提示词样式 | `styles.csv` |
| 生成的图片 | `output/txt2img-images`、`output/img2img-images`、`output/extras-images`，网格图和保存的图片在相邻目录（设置 → 图片保存路径） |
| 模型 | `models/`（或 `--model-ref`） |
| 用户扩展 | `extensions/` |
| **Remi 数据** | `remi-data/`：`jobs.db`（队列与历史）、`files.db`（文件索引、图库星标）、`uploads/`（你上传的图片，超过 `remi_upload_retention_days` 后清理）、`outputs/`（未被保存设置写入输出目录的结果，按同样规则清理）、`thumbs/`（缩略图缓存）、`networks/`（提取的预览图和占位图）、`prompt-tools.db`（提示词历史、收藏、补全使用频率）、`secret.key` 和 `revoked-sessions.json`（登录会话） |
| 自定义 CSS | `user.css`（存在时应用到网页界面） |
| 完成提示音 | 安装目录中的 `notification.mp3` |
| 浏览器本地 | 表单值（草稿）、外观、语言、面板尺寸：保存在浏览器的本地存储中。生成菜单中的“重置所有参数”可将表单恢复为默认值 |

迁移安装或 Docker 卷时请保留 `remi-data/`；删除它会清空队列、历史、图库星标和提示词工具数据，但不会删除 `output/` 中的图片。

## 键盘快捷键

命令面板（Ctrl/⌘+K）列出所有命令及其快捷键；图像编辑器中按 `?` 可查看编辑器自己的快捷键列表。

| 位置 | 按键 |
| --- | --- |
| 全局 | Ctrl/⌘+K 命令面板 · Alt+N 模型网络抽屉 |
| 生成 | Ctrl/⌘+Enter 生成（运行中：中止并立即重新生成；开启“让 [Ctrl + Enter] 恢复为仅中断生成”设置后只中止）· Alt+Enter 跳过 · Esc 中止（不在文本框和对话框内时） |
| 提示词编辑器 | Ctrl/⌘+↑ / ↓ 调整选中内容的权重 · Alt+← / → 移动标签 · Tab 或 Enter 接受补全 · Ctrl/⌘+Z / Y 撤销 / 重做 |
| 图像编辑器 | B 画笔 · E 橡皮擦 · G 填充 · M 矩形 · L 套索 · I 取色器 · H 抓手 · [ / ] 画笔大小 · 按住 Shift 擦除 · Ctrl/⌘+Z / Y 撤销 / 重做 · Ctrl/⌘+X 清空图层 · Ctrl/⌘+I 反转蒙版 · R 或 0 适应窗口 · 1 实际大小 · + / − 缩放 · F 最大化 · 空格+拖动 平移 · ? 帮助 |
| 查看器 | ← / → 上一张 / 下一张 · Home / End · + / − / 0 / 1 缩放 · S 下载 · T 平铺预览 · Esc 关闭 |
| 图库 | 方向键移动 · Enter 打开 · 空格选择（Shift 扩展选择）· F 星标 · Ctrl/⌘+A 全选 · Delete 移除 · Esc 清除选择 |
| 模型网络 | / 搜索 · 方向键移动 · Enter 应用 · I 元数据 · E 编辑 · Shift+F10 条目菜单 |
| 队列 | 空格拿起任务，↑ / ↓ 移动，空格放下，Esc 取消 |
| 后期处理 | Ctrl/⌘+Enter 处理 · Esc 中止 |
| 设置 | / 搜索 · Ctrl/⌘+S 应用 |

## 从 Gradio 界面迁移

Remi 的界面是全新的；生成引擎、模型、设置文件、样式、输出目录和 infotext 格式都与之前相同，你的文件和图片可以原样沿用。

| 旧界面 | Remi |
| --- | --- |
| 文生图和图生图标签页 | **生成**页，顶栏中切换文生图 / 图生图 |
| 图生图子标签（图生图、涂鸦、局部重绘、涂鸦重绘、上传蒙版重绘、批量处理） | **源图**卡片中的模式选择器；图像编辑器在舞台上 |
| 顶部的 checkpoint、VAE / 文本编码器和预设下拉框 | 顶栏中的**模型栏** |
| 快捷设置 | 顶栏的 ⚡ 快捷设置；在设置页中逐项固定 |
| 提示词下方的 Extra Networks 标签页 | **模型网络**抽屉（Alt+N）和**模型网络**页面 |
| “无限生成”右键菜单、Repeat Generate | 生成按钮旁的菜单：无限生成、连续生成多次 |
| Agent Scheduler 标签页和“Enqueue” | **队列**页面，以及生成按钮旁的**加入队列**（Ctrl+Shift+Enter）；队列设置保留，布局类设置见[其 README](extensions-builtin/agent-scheduler/README.md) |
| Prompt Queue 标签页 | **队列**页面 |
| Prompt All-in-One | 每个提示词下方的提示词工具栏（首次使用时自动导入原有历史和收藏）；各项旧功能的去向见[其 README](extensions-builtin/sd-webui-prompt-all-in-one-neo/README.md) |
| TagComplete | 内置自动补全（原有设置保留，未保留的设置及原因见[其 README](extensions-builtin/sd-webui-tagcomplete-neo/README.md)） |
| 括号检查、移动端布局 | 已内置 |
| PNG Info 标签页 | **图片信息**页面 |
| 设置标签页及其“默认值”和“操作” | **设置**页面；服务器操作位于“服务器与模型”。不再读取 `ui-config.json`：表单从默认值、界面预设和浏览器保存的值开始 |
| 扩展标签页 | **扩展**页面 |
| JoyCaption 和 Prompt Workshop 标签页 | 同名页面 |
| 模型合并（Checkpoint Merger）标签页 | **模型融合** 页面（控件相同；合并与保存在生成之后排队执行，显示进度并可取消） |
| `/file=` 链接 | 文件通过 id 提供（`/api/v2/files/{id}`） |
| `/agent-scheduler/v1/*`、`/prompt-queue/*` API | `POST /api/v2/jobs` 和 `/api/v2/queue/*`（[API 指南](docs/remi-webui/api.md)）；旧路由返回 `410 Gone` |
| `/tacapi/v1/*`、`/physton_prompt/*` API | `/api/v2/prompt-tools`（旧路由返回 `410 Gone`） |

**迁移旧队列。** 打开 **队列 → ⋯ → 从 Agent Scheduler 或 Prompt Queue 导入…**。导入器只读取旧数据、不做修改，可以安全地重复运行（已导入的任务会跳过）。Agent Scheduler 中待运行的任务会进入队列（队列会先暂停，便于你检查），已完成的任务连同图片进入历史，书签变为星标。按位置保存的脚本参数会根据当前面板命名；无法全部命名的任务会以“不可重放”状态保留并列在报告中：在对应表单中打开即可检查并重新排队。Agent Scheduler 数据库默认位于 `extension-data/agent-scheduler/task_scheduler.sqlite3`，或 `--agent-scheduler-sqlite-file` 指定的位置；Prompt Queue 的 `queue.json` 位于 `extension-data/sd-webui-prompt-queue/`。

**第三方扩展。** 只处理图像、没有界面的扩展通常可以继续使用。带 Gradio 界面（`ui()`、`on_ui_tabs`、JavaScript）的扩展需要移植：仍声明 `ui()` 的脚本会被禁用，导入 Gradio 的文件会被跳过，原因会显示在控制台和 `GET /api/v2/app`（`not_ported`）中；WebUI 的其余部分照常工作。移植方法见[扩展开发指南](docs/remi-webui/extension-guide.md)。为 `/sdapi/v1` 编写的 API 客户端在加上 `--api` 后可继续使用。

## 内置扩展

Remi 内置一组经过筛选、已全部移植到新界面的扩展：

- ADetailer Neo、NegPiP、Detail Daemon、Prompt Anchor、Prompt Blacklist、Post Processing Suite、Lock img2img Steps
- JoyCaption 和 Prompt Workshop（页面）、提示词工具（Prompt All-in-One 的后继）与标签补全、语言包
- Forge ControlNet 及其预处理器、ControlLLLite 和 IP-Adapter、MultiDiffusion、Image Stitch、Soft Inpainting、Never OOM、Sparse / Radial attention、编译和 PiD
- Remi 的 Anima 引导、Regional、FreeFuse、Artist Mixer、DAVE、Cross-Attention Boost、LoRA 分层权重与阶段调度、提示词缩放、分辨率、Dynamic Shift、Hires Guard、Spectrum 和预设

Agent Scheduler、Prompt Queue 和 Repeat Generate 已成为核心的队列功能。“内置”表示这些扩展与 Remi 一起版本化并受其测试覆盖，并不代表所有外部扩展都兼容。

## 安全与远程访问

默认情况下服务器只监听 `127.0.0.1`。

- **登录。** `--auth` / `--auth-file` 保护所有内容：网页界面、`/api/v2`、文件、WebSocket、`/internal` 和 API 文档。浏览器会看到登录页；会话有效期 30 天，重启后仍然有效，退出登录会将其吊销。脚本可以改用 HTTP Basic 凭据。同一客户端一分钟内登录失败十次后会被锁定一分钟。`/sdapi/v1` 在设置了 `--api-auth` 时使用它；否则使用 `--auth` 登录（脚本可用 HTTP Basic），仅监听本机时也是如此，因为同一台机器上的反向代理或隧道同样从回环地址连接。无法发送凭据的本地工具可以用 `--api-no-auth-on-localhost` 放行（来自本机且未经过代理的请求）。
- **对外开放服务器。** 请同时使用 `--listen`（或 `--server-name`）和 `--auth`，最好放在 TLS 反向代理之后，或使用 `--tls-keyfile` / `--tls-certfile`。不支持 `--share`。网络监听时，除非指定 `--enable-insecure-extension-access`，否则不能从界面安装和更新扩展；图库中的“在文件夹中显示”只对同一台机器上的浏览器有效，`--hide-ui-dir-config` 会移除所有在服务器上打开或定位文件夹的操作。
- **拒绝跨站请求。** 来自其他网站页面的写操作请求和 WebSocket 连接会被拒绝，恶意网页无法操控你的 WebUI。脚本和 API 工具不发送 `Origin` 头，不受影响；其他来源的浏览器前端需要用 `--cors-allow-origins` 放行。
- **拒绝 DNS 重绑定。** 仅监听本机的服务器只响应 `localhost` / 回环主机名（或 `--cors-allow-origins` 放行的主机）。本地反向代理必须以这些名称作为 `Host` 转发，或放行其公开来源。
- **文件**只能通过不透明 id 访问，上传内容会检查类型和大小（`remi_upload_max_mb`），其他请求体的大小上限是它的两倍，扩展 HTML、infotext 和元数据都以纯文本显示或经过净化。
- **扩展路由**（扩展添加到服务器的路由）同样经过跨站和主机检查；启用 `--auth` 时，它们还需要登录或 `--api-auth` 凭据（仅在使用 `--api-no-auth-on-localhost` 时本机 API 客户端免登录）。
- **启动警告**会指出有风险的组合：无登录即可被访问、`--cors-allow-origins=*`、不安全的扩展访问、未使用 TLS 的登录。威胁模型和对外开放的检查清单见 [docs/remi-webui/security.md](docs/remi-webui/security.md)。
- **对外网络请求**只发生在你使用或配置的功能中：扩展页面中的扩展索引、你配置的翻译服务（若为 OpenAI 兼容服务，“AI 撰写提示词”时你输入的描述也会发送给它），以及部分功能首次使用时的模型下载（CLIP Modulation、JoyCaption、ControlNet 预处理器、ADetailer）。

## 故障排查

**浏览器显示“The web interface is not built”。** 启动器运行时未找到 Node.js、构建失败，或构建被跳过（`--skip-install`、`--skip-prepare-environment`、`REMI_SKIP_WEBUI_BUILD`），控制台会说明是哪一种。Node.js 版本过旧时会提示“Node.js vX is too old to build the web interface”，构建通常会失败。安装 Node.js 20.19+ / 22.12+ 后重启，或在其他机器上构建 `webui/dist` 再复制过来（[网页界面的构建](#网页界面的构建)）。

**构建失败**（`pnpm install` 或 `pnpm build` 报错退出）。在 `webui/` 中运行 `pnpm install --frozen-lockfile && pnpm build` 查看完整信息。使用代理时，请为 npm/pnpm 配置代理（`HTTPS_PROXY`）。服务器仍会启动，期间 API 可用。

**更新后页面看起来还是旧的。** 刷新页面；应用发现新的构建时也会自动刷新一次。如果仍然如此，说明构建没有运行（见上文）。

**403 “Unknown host refused”。** 你通过其他主机名（局域网名称、反向代理、隧道）访问了仅监听本机的服务器。请使用 `--listen` / `--server-name`，或用 `--cors-allow-origins=https://your.host` 放行该名称，或让代理发送 `Host: localhost`。

**403 “Cross-site request refused”。** 其他来源的页面（独立前端、用户脚本站点）发送了写请求。用 `--cors-allow-origins` 放行该来源。在反向代理之后，请转发公开的主机和端口（`X-Forwarded-Host`、`X-Forwarded-Port`、`X-Forwarded-Proto`）。

**没有实时预览或进度，顶栏的连接指示为红色。** WebSocket（`/api/v2/ws`）被拦截。反向代理必须转发 WebSocket 升级（nginx：`proxy_http_version 1.1; proxy_set_header Upgrade $http_upgrade; proxy_set_header Connection "upgrade";`）。同时检查设置中是否启用了实时预览。

**在反向代理的子路径下页面空白。** 使用与代理路径一致的 `--subpath /your/prefix` 启动。

**端口被占用。** 未指定 `--port` 时，服务器会依次尝试 7860、7861 …… 直到 7879；指定 `--port` 时该端口被占用则启动失败。请在控制台中查找 `Running on local URL:`。

**某个扩展的面板或页面不见了。** 查看控制台：声明 Gradio `ui()` 而非 `params()`、或导入 Gradio 的脚本会被禁用并输出说明（也会列在 `GET /api/v2/app` 的 `not_ported` 中）。禁用的扩展在扩展页面中管理；更改在“重新加载界面”后生效。

**提示“Extension access is disabled”。** 服务器正在网络上监听；如果你信任所有能访问它的人，可以加上 `--enable-insecure-extension-access` 启动。

**表单保留了奇怪的值。** 表单值按浏览器保存。使用生成菜单中的“重置所有参数”，或点击单个值旁的重置圆点。

**登录状态无法保持。** 使用 `--tls-*` 或在 HTTPS 之后时，会话 cookie 带有 `Secure` 标记，请访问 HTTPS 地址。cookie 的作用路径受 `--subpath` 限定。

**显存不足或生成缓慢。** 这些属于引擎问题，与 Forge Neo 相同：尝试 Never OOM 面板、“低精度扩散”中的低比特 dtype、更小的批次，以及原生 PyTorch attention。

## 文档

- [文档索引](docs/remi-webui/README.md)
- [Remi Anima 生成与参数指南](docs/remi-anima-generation-guide.zh-CN.md)
- [API 指南（REST、WebSocket、/sdapi）](docs/remi-webui/api.md)
- [扩展开发指南](docs/remi-webui/extension-guide.md)
- [WebUI 架构](docs/remi-webui/architecture.md)与 [Anima 功能接入点](docs/remi-architecture-notes.md)
- [吸收 Forge Neo 更新](docs/remi-webui/upstream-absorption.md)
- [Anima 生成优化调研](docs/research/anima-generation-optimizations.md)
- [Forge Neo Wiki](https://github.com/Haoming02/sd-webui-forge-classic/wiki) · [模型下载](https://github.com/Haoming02/sd-webui-forge-classic/wiki/Download-Models) · [推理参考](https://github.com/Haoming02/sd-webui-forge-classic/wiki/Inference-References)

（开发者文档目前为英文。）

## 兼容性说明

- Forge Neo 已移除的训练功能、旧模型家族和旧版 WebUI 组件不会在 Remi 中恢复。
- 为 Gradio 界面编写的扩展需要移植；只处理图像的扩展和 `/sdapi/v1` 客户端通常无需修改。
- FlashAttention 等 CUDA 专用组件并非所有平台都可用。
- 实验性引导方法可能同时改变画面风格和质量；对比时请保留固定 seed 的基准图。
- Torch compile（整模型预设和 **Anima per-block**）在 NAG、NegPiP、Regional、FreeFuse、Artist Mixer、DAVE 或 Cross-Attention Boost（各自生效区间内）以及 Sparse Attention 启用时，相关步以未编译方式运行，确保这些功能不会被缓存的计算图静默忽略；这类运行不会获得编译加速。
- **Anima FlowMatch** 只对 flow-matching 模型有意义；在其他模型上（例如从 Anima Flow 采样器切换模型后仍保留该选项），会回退为 Simple 调度。
- 真实 checkpoint 的生成结果才是最终兼容性检验。启动测试或单元测试不能证明每种 checkpoint 与量化格式的出图质量。

## 问题反馈与贡献

提交问题前，请在当前 `remi` 分支上复现，并禁用第三方扩展。请附带：

- Remi commit hash
- 操作系统、浏览器、GPU/后端、Python 和 PyTorch 版本
- checkpoint 家族与量化格式
- 启动参数和相关生成参数
- 完整控制台 traceback、界面问题的浏览器控制台输出，或最小复现

Remi 特有问题请提交到 [Remi Issues](https://github.com/beautifulrem/sd-webui-forge-classic/issues)。仅与上游有关的问题，请先在 [Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo) 中复现后再向上游反馈。

贡献代码时请保持功能隔离，不改变非 Anima 行为；修改共享采样或模型加载路径时请增加回归测试，并记录界面与运行时的冲突语义。`tools/check.sh` 会运行全部检查（Python 与前端测试、lint、类型检查、构建）。

## 上游与致谢

Remi 建立在 [AUTOMATIC1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui)、[lllyasviel](https://github.com/lllyasviel/stable-diffusion-webui-forge)、[Haoming02](https://github.com/Haoming02/sd-webui-forge-classic)、[ComfyUI](https://github.com/Comfy-Org/ComfyUI) 以及内置扩展和算法作者的工作之上。原始许可证和声明保留在各自组件中。

## 许可证

主项目采用 [GNU Affero General Public License v3.0](LICENSE)。内置扩展和 vendored 组件可能附带其他兼容许可证，请查阅对应目录中的 `LICENSE` 文件。
