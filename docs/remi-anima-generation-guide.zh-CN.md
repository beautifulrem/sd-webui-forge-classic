# Forge Neo Remi 功能与 Anima 生成指南

> 适用分支：`remi`  
> 代码核对基线：`8169d6ca`（2026-08-09）  
> 本文中的“推荐”是保守起步值，不是对所有模型、LoRA 和提示词都最优的固定答案。

## 先看结论

1. 第一次测试一个 Anima checkpoint 时，先关闭所有 Remi 质量增强开关，只建立官方基线。
2. Base/Aesthetic 从约 1MP、30–50 步、CFG 4–5 开始；Turbo 从 8–12 步、CFG 1 开始。
3. 一次只加入一个变量，并保持 checkpoint、prompt、seed、尺寸、步数和 CFG 不变做 A/B。
4. `Beta57`、Dynamic Shift 和锁定 `Anima FlowMatch` 的采样器是不同的时间步路径，不能同时起作用。
5. NAG、SMC、FDG、DCW、CNS、Momentum 都是针对特定问题的工具，不应全部叠加。
6. Spectrum、`torch.compile` 是速度功能，不会天然提高画质；PiD 是独立的 4× 后处理器。

目前仓库已完成 CPU 单元测试、组合回归和静态检查，但还没有完成覆盖 Base、Aesthetic、Turbo 与多张显卡的固定 seed GPU 画质矩阵。因此本文把“代码已经实现”和“画质已经普遍证明”严格分开。

## GUI 内置推荐预设

打开 txt2img 或 img2img 后，在页面下方展开 `Anima Remi Presets`，选择界面已预选的 `Official Base/Aesthetic — Safe (Recommended)`，然后点击 `Apply recommended preset`。预设不会在启动时强行改写当前界面，只有点击后才应用。

它会一次设置 `ER SDE`、`36` 步、CFG `4.5`、`Automatic` scheduler，并把 Guidance、Dynamic Shift、Spectrum、Regional、Artist Mixer、FreeFuse、PiD、Hires Guard、LoRA Stage/Layer、Prompt Rescale、分辨率随机化和 Hires/Refiner 等可选栈关闭；这些模块内部的数值也会恢复为本文推荐起点。Torch Compile 回到 `Automatic`。

预设不会清空正/负提示词、Regional prompt、Artist 名称与权重、LoRA 名称、FreeFuse trigger phrase、PiD 模型选择或本地 adapter/encoder 路径。这样可以安全建立采样基线，而不破坏用户正在编辑的核心创作内容。它面向 Anima Base/Aesthetic；Turbo 请在应用后再改为约 `10` 步、CFG `1.0` 和 `Euler`/`Anima Flow Euler`。

## 三套可以直接起步的配置

### 1. Anima-Base / Anima-Aesthetic 稳定基线

| 项目 | 建议 |
|---|---|
| 尺寸 | `1024×1024`，或同量级的 64 倍数长宽比 |
| Steps | `36–40` |
| CFG | `4.5` |
| Sampler | `ER SDE`；想增加变化可试 `Euler` |
| Schedule type | 先用当前模型/Forge 的普通默认路径 |
| Anima Guidance & Corrections | 关闭 |
| Spectrum / Dynamic Shift / Compile | 关闭 |
| Hires | 关闭；确认底图正常后再开 |

Base 更自由、更多样，也更依赖画师和质量标签；Aesthetic 默认更稳定。Aesthetic 不建议同时在正向和负向堆叠大量 `score_*` 标签。

### 2. Anima-Turbo 快速迭代

| 项目 | 建议 |
|---|---|
| 尺寸 | `1024×1024` 或接近 1MP |
| Steps | `10`（在 `8–12` 内调整） |
| CFG | `1.0` |
| Sampler | `Euler` 或 `Anima Flow Euler` |
| Guidance | 先关闭；负面提示控制不足时再单独试 NAG |
| Spectrum | 通常关闭：少步数下可跳过的实际前向很少 |

Turbo 的稳定性和默认风格更强，但多样性较低。不要把 Base/Aesthetic 的 CFG 4–5 直接套给 Turbo。

### 3. 厚涂、绘画感和更强调低噪声纹理

| 项目 | 建议 |
|---|---|
| Model | Base 或 Aesthetic |
| Steps / CFG | `36–45` / `4.0–4.5` |
| Sampler | `ER SDE` 或 `Euler` |
| Schedule type | `Beta57` |
| 其他 guidance | 第一次对比时全部关闭 |

`Beta57` 更强调低噪声阶段，可能改善绘画和厚涂纹理，但它不是“通用质量增强”。它会改变纹理取向，线稿、平涂和某些 LoRA 未必更好。

## 官方基线和提示词

[Anima 官方模型卡](https://huggingface.co/circlestone-labs/Anima)给出的基线是：

- Base/Aesthetic 支持约 `512²–1536²`，通常使用 30–50 步、CFG 4–5；
- Turbo 使用 CFG 1、8–12 步；
- 官方常用采样器包括 `ER SDE`、`Euler a`、`DPM++ 2M SDE` 和 `Euler`；
- `Beta57` 对写实倾向或厚涂纹理可能有帮助；
- 标签使用小写和空格，只有 `score_*` 保留下划线；
- 画师名前必须加 `@`；
- 纯自然语言应写得足够具体，多人物时同时描述每个人的外观。

建议先让提示词本身工作，再用生成期增强修正具体问题。高级 guidance 无法弥补缺少主体、姿势、视角或人物区分信息的短提示词。

## Remi 分支新增能力地图

### 直接影响 Anima 采样或画质

- `Beta57` scheduler；
- `Anima Flow Euler`、`Anima Flow UniPC2`、`Anima Flow PC3`；
- Skimmed CFG、SMC-CFG、FDG、Guidance Active Range；
- NAG、Momentum Guidance、DCW、ER-SDE CNS、CLIP modulation；
- Spectrum hidden-feature forecasting 与 SEA；
- Dynamic Shift；
- Anima Hires Guard、PiD 4× 后处理；
- Regional Conditioning、FreeFuse、Artist Mixer；
- LoRA stage scheduler 与 layer weight；
- Anima prompt schedule rescaler 和分辨率工具。

### 通用工作流功能

Remi 还把 ADetailer-Neo、Detail Daemon、Repeat Generate、Agent Scheduler、NegPiP、Prompt All-in-One Neo、TagComplete Neo、Prompt Workshop、Prompt Blacklist、JoyCaption、Postprocess Suite、语言包等作为内置扩展提供。它们改善修脸、批量任务、提示词编辑、标签补全、反推和输出处理，但不会自动让 Anima 的采样数学更好。

不要在第一次基线测试中同时启用 ADetailer、Detail Daemon、Hires 和 PiD。它们都可能触发额外处理，使问题来源难以判断。

## 采样器和 scheduler 怎么选

| 目标 | 建议入口 | 说明 |
|---|---|---|
| 官方风格基线 | `ER SDE` | 中性、平涂、线条较锐 |
| 更柔和或更自由 | `Euler a` / `Euler` | `Euler a` 更柔，`Euler` 变化更大 |
| 官方 Diffusers flow 路径 | `Anima Flow Euler` | 最容易理解和排错；支持 Momentum |
| flow predictor-corrector | `Anima Flow UniPC2` | 保持默认 `bh2`，先不要开 thresholding |
| 更强的端点修正实验 | `Anima Flow PC3` | 默认 gamma `1.0`、tolerance `0.005`；额外求值更多 |
| ER-SDE 频谱噪声实验 | `Anima ER SDE CNS` | 在 `Anima Guidance & Corrections` 中控制强度 |
| 2–3 个主体 LoRA 空间分离 | `Anima FreeFuse Euler` | 开启 FreeFuse 后会自动选择 |

三个 `Anima Flow *` 采样器会把 scheduler 锁定为 `Anima FlowMatch`。因此：

- 选择 `Beta57` 不会改变它们；
- Dynamic Shift 的 scheduler override 也会被忽略；
- 如果要比较 `Beta57` 或 Dynamic Shift，请改用 `ER SDE`、`Euler` 等普通 K-diffusion 采样器。

UniPC2 和 PC3 第一次使用应保留界面默认值。只有在固定 seed 对比中确认需要后，才调整：

- UniPC2：`bh2`、早期禁用 corrector 数 `0`、dynamic thresholding 关闭；
- PC3：gamma `1.0`、tolerance `0.005`。若修正过强，先把 gamma 降到 `0.5–0.8`，不要同时改 tolerance。

## Anima Guidance & Corrections

整个面板默认关闭。选择了专用采样器但不展开该面板时，采样器仍可使用自己的代码默认值；只有需要修改 CNS、UniPC2 或 PC3 参数时才必须启用面板。

### Standard / preserve existing

这是日常默认。它保留已有 CFG combine，仅按需叠加 Skim、NAG、DCW、Momentum 或 Active Range。

### NAG attention guidance

用途：负面提示控制弱、Turbo CFG 1 下需要更明确排除某些内容，或普通 CFG 已难以继续提高而不烧图。

Turbo 建议起点：

| 参数 | 起点 |
|---|---|
| NAG scale | `2.0` |
| Normalization tau | `2.5` |
| Blend alpha | `0.5` |
| Start / End | `0.0 / 0.5` |

Base/Aesthetic 可先保守试 `scale 1.0–1.5`、`alpha 0.25–0.4`、`tau 2.5`、范围 `0.0–0.35`。这是 Remi 的保守 A/B 起点，不是官方参数。

注意：

- NAG 需要正负条件在同一个 GPU batch 中。低显存导致 Forge 拆分分支时，它会明确记录为 inactive；
- NAG 会强制真实的正负分支和 Spectrum actual forward，速度与显存开销都会增加；
- 与 Regional/Artist Mixer 同时使用时，NAG 只引导基础 prompt 路径，合成的区域/画师 context 会保持隔离。

### Momentum Guidance

用途：在 Euler flow 轨迹上利用历史速度 EMA，使跨步方向更平稳。它不增加模型前向，但会改变轨迹。

建议从代码默认值 `strength 0.5`、`EMA 0.6` 做固定 seed A/B；若风格被拉得太强，先把 strength 降到 `0.2–0.35`。

硬限制：

- 只支持 `Euler`、`Anima Flow Euler`、`Anima FreeFuse Euler`；
- SMC-CFG 或 FDG 开启时自动禁用；
- 不用于 UniPC2、PC3、Heun、DPM2 等含中间求值的采样器；
- FreeFuse 的收集阶段结束后会重置 Momentum 状态。

### Skimmed CFG anti-burn

用途：Base/Aesthetic 在较高 CFG 下出现过饱和、发黑、发白或局部“烧焦”时，仅限制异常值。

建议：

- `Skimming fallback CFG = 2.5`；
- `Full skim negative = Off`；
- `Disable flipping filter = Off`；
- Start/End 先用 `0.0/1.0`。

它是比直接堆另一种 CFG controller 更适合先试的防烧修正。Turbo CFG 1 通常没有必要开启。

### SMC-CFG

用途：实验性地自适应修正 cond/uncond 的控制误差。建议只在 Standard 基线出现明显结构不稳时单独测试。

起点：`alpha 0.1–0.2`、`lambda 3–5`；代码默认是 `0.2 / 5.0`。参数越高并不等于越好，出现轮廓僵硬或过强对比时先降低 alpha。

SMC-CFG 会占用 CFG combine，并自动禁用 Momentum。不要同时选择 FDG；二者在同一个下拉框中本来就是互斥模式。

### FDG (experimental)

用途：把 guidance residual 分成低频结构和高频细节，并单独提高高频 guidance。它尚无 Anima GPU 量化结论。

建议只做研究 A/B：先用 `detail guidance 1.5–2.0`，其余增强关闭。代码默认是 `2.0`。

硬限制：

- 与 SMC-CFG 互斥；
- 自动禁用 Momentum 和 DCW；
- 选择 `Anima ER SDE CNS` 时 FDG 会被禁用；
- 不建议再叠加频域锐化或强后处理。

### Limit CFG to a sampling range

用途：只在采样中段使用 CFG，减少早期或末期过强引导。Base/Aesthetic、CFG 4–7 才值得试；Turbo CFG 1 基本没有收益。

建议从 `0.10–0.80` 做 A/B。范围外退回 conditional/CFG 1 行为。该范围是 flow-aware 的归一化采样进度，不是从其他模型照搬的 sigma 数值。

### DCW latent bias correction

用途：修正跨步 latent 的低频偏移，例如整体亮度、色彩或大面积背景逐步漂移。

Anima 起点就是代码默认值：

- lambda `-0.015`；
- schedule `one_minus_sigma`；
- bands `LL`。

只在确实看到低频漂移时开启。优先保持 `LL`，不要为了“更多细节”切到 `all`；DCW 不是锐化器。FDG 开启时 DCW 会被禁用。

### Anima ER SDE CNS

用途：保留 ER-SDE 求解器，只重着色每一步的随机噪声频谱。

选择 `Anima ER SDE CNS` 后，先用 `strength 0.5` 与普通 ER SDE 对比，再试默认的 `1.0`。`0` 等于 stock ER-SDE 噪声，`1` 是完整 Anima 校准重着色。首次使用会下载并校验约 6 KiB 的校准文件。

CNS 不作用于 Flow Euler/UniPC2/PC3；FDG 与 CNS 不同时运行。

### CLIP modulation guidance

用途：沿“正向语义方向 − 负向语义方向”调制 Anima block，更适合目标明确的质量、解剖或风格方向，不是通用必开项。

第一次可保留：weight `3.0`、blocks `0..-1`、默认正负方向。若主体身份或画风漂移，先把 weight 降到 `1–2`。自动模式首次会下载并哈希校验约 163 MiB adapter 和 235 MiB CLIP-L encoder。

## Dynamic Shift

Dynamic Shift 在普通 flow-compatible K-diffusion sampler 中改变步骤分布：高 shift 偏早期构图，低 shift 偏后期细节。

建议起点：

- 通用构图优先：`5.0 → 1.5`，`Cosine`；
- 温和版本：`3.0 → 1.0`，`Linear`；
- img2img 已固定构图时，才偶尔尝试反向 ramp。

它默认关闭，并且会替代 Schedule type 与普通 Shift 对 sigma schedule 的作用。不要和 `Beta57` 当作两个同时生效的增强；启用 Dynamic Shift 时它自己的 schedule override 优先。三个锁定的 `Anima Flow *` sampler 会忽略它。

## Spectrum 和 torch.compile

### Spectrum Integrated

用途：预测 Anima 最终 DiT 层之前的 hidden feature，从而跳过部分完整 block 前向。它是速度/显存权衡，不是质量增强。

推荐规则：

- 8–12 步 Turbo：通常关闭；
- 30–50 步 Base/Aesthetic、同配置连续出图：可以开启；
- 第一次保留全部代码默认值：weight `0.25`、degree `4`、regularization `0.1`、window `2`、growth `0`、warmup `6`、stop `0.9`、tail actual `3`、history `10`、`Window + Conservative`；
- SEA 适合反复使用相同 sampler/CFG/分辨率，第一次运行负责校准；
- `Legacy / fastest` 是明确的画质风险选项，不作为推荐；
- `Strict` 当前等同全部 actual forward，主要用于排错。

NAG、Momentum、Regional、FreeFuse、PC3 probe 等需要真实模型状态的路径会强制 Spectrum actual forward，因此开启面板不代表一定获得加速。

### Torch Compile Integrated → Anima per-block

用途：逐个编译 28 个 Anima transformer block，降低整模型 graph break 风险。只改善后续运行速度。

建议仅在以下条件同时满足时开启：

- 已经确定 prompt/LoRA/采样配置；
- 会重复相同分辨率和 batch size；
- Triton/Inductor 与系统编译工具链可用；
- 能接受第一次生成明显变慢。

频繁改尺寸、batch 或模型时保持 `Automatic`/不编译更省事。懒编译失败会记录到当前图片元数据并持续回退 eager；Windows 缺少 C++ 编译器时会提前禁用。

Spectrum 与 compile 可以技术上组合：前者减少完整前向次数，后者加速仍然执行的 block。但收益不相加，必须用本机计时验证。

## 区域、画师与多 LoRA

Remi 会按 checkpoint 实际结构识别 28 层和 40 层/2.9B Anima。完整的旧 28 层 LoRA 只会在当前模型明确为 40 层时做保守重映射；原生 40 层 LoRA 和无法判断来源的部分层 LoRA 保持原样。GUI 中的 `0-39` 全层默认在 28 层模型上会自动忽略越界层。

### Anima Regional Conditioning

适合把最多三个独立 prompt 路由到矩形区域。保守起点就是默认值：

- blocks `0-39`（28 层模型会自动忽略 28-39）；
- progress `0.0–0.65`；
- feather `0.03`；
- base preserve `0.15`；
- region strength `1.0`。

先从一个区域开始，再增加第二个。区域重叠会做归一化，不依赖行顺序。它和 NAG 可同时启用，但 NAG 不会对合成 regional context 再做负向引导。Regional 与 FreeFuse 是硬冲突。

### Anima Artist Scheduled Mixer

画师标签必须带 `@`，建议每行一个画师。推荐保留插件默认：

- 优化预设 `平衡`；
- 组合模式 `输出平均`；
- 融合模式 `插值融合`；
- 全局强度 `0.70`。

风格太强时先切 `保真增量` 或降低全局强度，不要先增加更多画师。Artist Mixer 与 FreeFuse 是硬冲突；与 Regional 虽能组合，也应先分别建立基线。

### Anima LoRA Stage Scheduler

只在 LoRA 影响阶段不合适时开启，例如人物 LoRA 破坏构图、风格 LoRA 太早锁死布局。建议：

- 保持“根据 Shift 自动设置介入时间”开启；
- 先用内置构图、人物、画风模板；
- LoRA 本体先用中等权重；
- Hires 先继承底图设置，需要时再分离。

### Anima LoRA/LoKR Layer Weight

只在确定某个 LoRA 的层分布有问题时开启。实用但非硬规则的层含义：

- `0–9`：结构、姿态、构图；
- `10–18`：人物、服装和主要特征；
- `19–27`：画风、光影和细节。

建议先降低不希望受影响的层，而不是把目标层推到很高。可从示例 `0–9=0.7, 10–18=1.0, 19–27=0.7` 起步。Stage Scheduler 与 Layer Weight 可以组合，但同一 batch 内提示词若给同一 LoRA 提供冲突规则，生成会主动中止而不是静默串状态。

### Anima FreeFuse (multi-LoRA)

仅用于 2–3 个独立主体 LoRA 的空间隔离。开启后自动改用 `Anima FreeFuse Euler`，并进行同噪声两阶段采样。

第一次保持默认：collect step `3`、block `18`、top-k `0.10`、temperature `300`、background scale `0.95`、balance `15`、feather `1`、routing strength/end `1.0/1.0`、wrong-concept suppression `6.0`、own-concept boost `1.0`。

硬要求：

- 每个启用项都要填写匹配已加载文件名/Stem 的 LoRA 名；
- trigger phrase 必须逐字出现在每一张图的正向 prompt；
- `S-churn = 0`；
- 不与 Regional、Artist Mixer 或 Refiner checkpoint switch 同跑；
- LoRA stage/layer weight 与 modulation 可以组合；
- Spectrum 会被强制实际前向，因此不要期待同时获得缓存加速。

## 分辨率、Hires 和 PiD

### Anima Resolution Helper

- 日常 txt2img 用 Standard preset；
- High-res preset 更适合有意识的直接高分辨率实验或 Send to img2img 的最终尺寸；
- Randomize 每次 Generate 只选择一个分辨率，整个 batch 共用；
- Send to img2img 会根据图片 PNG 元数据把 Resize-to 对齐到 64 倍数并尽量保持比例；必须停留在 `Resize to` 页签。

High-res preset 与 Hires Guard 的默认 Base 上限不是同一个目标：Balanced Guard 会把大于 1.2MP 的直接 txt2img base 向下限制。若有意直接生成 2MP 级底图，应把 Guard 改成 `Report only` 或审慎设置 Custom，而不是以为两个面板会自动协商。

### Anima Hires Guard

只要对 Anima 开 Hires，建议同时启用 Guard：

- Policy：`Clamp unsafe values`；
- Preset：`Balanced`；
- Align to 16 px：开启；
- 对应限制：base `1.2MP`、Hires `2.1MP`、每轴最多 `1.75×`。

低显存用 `Low VRAM`（0.9/1.5MP、1.5×）；明确有余量再试 `Detail`（1.4/2.36MP、2×）。第一次可以用 `Report only` 观察会发生什么，但它不会保护显存。

### PiD Integrated

PiD 是当前 latent 之后的 4-step、4× pixel diffusion 后处理，不是普通 VAE decode。推荐默认关闭；用于最终候选图时：

- 必须关闭 Hires fix；
- Degrade Sigma 先用 `0.0`，需要更多重绘时逐步试 `0.05–0.15`；
- Color Correction 保持开启；
- 预先安装 PiD checkpoint、VAE 和 Gemma2 2B IT ELM；
- 预留明显的时间、显存和输出尺寸空间。

PiD 权重仅允许非商业研究/评估，使用前必须自行核对其模型许可。它与普通 Anima 模型权重和生成输出的许可不是同一件事。

## Img2img 提示词调度

如果 prompt 包含 `[from:to:when]`，从 txt2img 发送到较低 denoise 的 img2img 时，启用 `Anima Prompt-Schedule Rescaler` 的 Auto-rescale，并把 Source steps 设置为原 txt2img 步数。

普通 prompt、`[a|b]` 交替和 `(tag:weight)` 不需要开启。PC3/UniPC/Heun 等每个主步含多次模型调用，精确切换位置可能偏一小步；需要时手动微调。

## 兼容性速查

Remi 会在 GUI 中采用“最后选择的功能获胜”：冲突项会被自动关闭并锁定，关闭当前获胜项后再全部解锁。API/CLI/PNG 参数恢复可以绕过 GUI，因此采样前还会再次运行同一套规则，并把自动替换、禁用或 fallback 原因写进 infotext。Headless 模式没有点击顺序时采用固定安全优先级：专用 sampler/controller 优先，FreeFuse 优先于 Regional、Artist Mixer 和 Refiner。

| 组合 | 结论 |
|---|---|
| `Beta57` + `Anima Flow Euler/UniPC2/PC3` | Flow sampler 锁定 scheduler，Beta57 不生效 |
| Dynamic Shift + `Anima Flow *` | 被 scheduler lock 忽略 |
| Dynamic Shift + Beta57 | Dynamic Shift override 实际接管 schedule；不要当作双重增强 |
| SMC + FDG | 互斥模式 |
| Momentum + SMC/FDG | 自动禁用 Momentum |
| Momentum + UniPC2/PC3/Heun/DPM2 | 不支持，自动禁用 |
| FDG + DCW | 自动禁用 DCW |
| FDG + CNS | 自动禁用 FDG |
| FreeFuse + Regional | 硬冲突，任务报错 |
| FreeFuse + Artist Mixer | 硬冲突，任务报错 |
| FreeFuse + Refiner checkpoint switch | 不支持 |
| NAG + Regional/Artist | 可运行；NAG 只作用基础 prompt 路径 |
| PiD + Hires fix | PiD 跳过并写入元数据 |
| Spectrum + NAG/Momentum/Regional/FreeFuse/PC3 probe | 相关步骤强制 actual，速度收益下降或消失 |
| Stage Scheduler + Layer Weight | 支持；batch 内冲突规则会主动中止 |

## 推荐调参顺序

1. 先确定模型版本、prompt、尺寸、普通 sampler、steps、CFG。
2. 只处理一个最明显的问题：烧图用 Skim；负向控制用 NAG；低频漂移用 DCW；ER-SDE 纹理实验用 CNS。
3. 再比较采样路径：普通 ER SDE/Euler、Beta57、Flow Euler、UniPC2 或 PC3，每次只换一个。
4. 多人物先用更清楚的 prompt；仍混淆时在 Regional、Artist Mixer、FreeFuse 中选择一个主方案。
5. 底图稳定后才加 Hires、ADetailer/Detail Daemon 或 PiD。
6. 最后再加 Spectrum、compile 等速度优化，并重新做相同 seed 回归。

若结果变差，按相反顺序关闭。PNG infotext 会记录实际启用、自动禁用、fallback 或 inactive 的 Anima 参数，应先看元数据再判断“开关没有效果”。

## 进一步文档

- [Anima 生成期优化技术调研](research/anima-generation-optimizations.md)
- [Anima Guidance & Corrections](../extensions-builtin/sd_forge_anima_guidance/README.md)
- [Spectrum vNext](../extensions-builtin/sd_forge_spectrum/README.md)
- [Anima FreeFuse](../extensions-builtin/sd_forge_anima_freefuse/README.md)
- [Anima Regional Conditioning](../extensions-builtin/sd_forge_anima_regional/README.md)
- [Anima LoRA 阶段介入](../extensions-builtin/anima-lora-stage-scheduler/README_zh.md)
- [Anima LoRA/LoKR 分层权重](../extensions-builtin/anima-lora-layer-weight/README_zh.md)
- [Anima 画师串调度混合](../extensions-builtin/anima-artist-scheduled-mixer/README_zh.md)
