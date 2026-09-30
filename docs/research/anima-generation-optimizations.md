# Anima 生成期优化技术与 Forge Neo 移植调研

> 调研日期：2026-08-09
> 对照分支：`remi`，基线提交 `d940ce41`
> 范围：只评估生成期质量、提示词/空间控制、采样、加速和高分辨率后处理；训练技术不作为本报告的主要对象。

## 结论

### Forge Neo 落地状态（2026-08-09）

本报告中进入移植队列的项目已按默认关闭、生成级隔离和明确互斥的原则集成到 `remi` 工作区：

- NAG 通过 Anima 原生 attention modifier chain 接入，不覆盖 Regional、Artist Mixer 或 FreeFuse 的 wrapper；合成的 Regional/Artist context 与 NAG 隔离；低显存导致正负分支被拆分时，正向调用用该层 K/V 投影负向 prompt context 继续生效，只有无法取得负向 context 时才记录 inactive；
- Momentum Guidance 在 flow velocity 空间维护每次生成独立的 EMA，按固定参考实现先对上一步 EMA 外推 `v + α·(v − m_prev)` 再更新 EMA（此前先更新 EMA 再外推，等效强度只有 α·β），限于单次评估的 Euler/Anima Flow Euler/FreeFuse Euler，FreeFuse probe 后会重置状态；
- Anima per-block compile 作为现有 Compile 内置插件的互斥预设，调用期临时替换 block，并在 `finally` 中恢复；Windows 先预检 C++ 编译器，Dynamo/Inductor 懒编译失败则回退 eager；attention modifier 可提供 `active(transformer_options)`，NAG 在 sigma 窗口外会被从该次调用中剔除，因此只有窗口内的步骤回退 eager；
- FDG 作为实验 CFG controller，使用纯 PyTorch 两层频带分解，不增加 Kornia 硬依赖，并与 DCW/CNS 显式互斥；低频段使用 CFG，`detail guidance` 是高频段的绝对 scale（同参考实现 `strength_high`），默认 `12`（原默认 `2` 低于 Anima 常用 CFG 4–5，反而削弱高频引导）；
- CFG-Zero* 依论文独立实现为同一下拉框中的 CFG combine 模式（与 SMC/APG/FDG 天然互斥，禁用 Momentum）：逐样本最优缩放 s* 在 Forge 的 `x - denoised = sigma * v` 残差上计算，可选 zero-init 前 N 步（默认 1）；zero-init 作为最外层 CFG hook，能看到 active range 窗口外的步骤，且只在 pass 从纯噪声（调度最大 sigma）开始时生效，Hires/img2img pass 不会被置零；
- APG（arXiv:2410.02416）作为同一下拉框中的 CFG combine 模式：按论文 Algorithm 1（亦见 Apache-2.0 的 diffusers `guiders/adaptive_projected_guidance.py`）在去噪预测 x0 空间计算，`eta` 加权平行分量、`norm threshold` 限制每个样本更新的 L2 范数、反向 momentum 通过 stateful guidance 注册表按 pass 重置；
- TCFG（arXiv:2503.18137）作为同一下拉框中的无参数 CFG combine 模式：逐样本用 `[uncond; cond]` 的 2×2 Gram 矩阵闭式求第一右奇异向量，只保留无条件残差在其上的分量后做普通 CFG，在 `x - denoised = sigma * v` 残差上计算，无状态、无额外前向；
- SMC-CFG 的 switching gain 按样本计算（参考实现对整个 batch 求均值，batch 1 时等价），同 batch 图片不再互相影响；
- Guidance Active Range 的起止值是模型去噪时间的比例，经 Forge 的 `percent_to_sigma` 换算为 sigma（Anima shift 3：0.5 对应 sigma 0.75），与 scheduler 无关，也不是步数比例（Beta57/30 步时 end 0.5 只覆盖 13 步）；NAG 的 start/end 同此约定；
- 现有 PiD 后处理器修复了类级结果泄漏和多批次 prompt/seed 错位，并在 GUI 中明示 Hires 互斥与权重许可限制。

目前已完成 CPU 单元/组合回归、Ruff、`compileall` 和 `git diff --check`；仍需在含 Anima 权重的 NVIDIA 环境执行报告末尾的 GPU 固定 seed 矩阵，才能对画质收益和编译加速比例作实证结论。

当前 `remi` 并不是“缺少 Anima 优化”的状态。官方明确推荐的 beta57、原生 flow Euler 路径、ER-SDE、控制条件、区域提示、LoRA 调度、缓存/预测、FreeFuse 和高分辨率防护等主要能力已经内置。继续盲目预装 ComfyUI 节点，更多会造成重复 patch、不可组合的 attention hook 和难以解释的 UI，而不是提高质量。

本报告最初建议的推进顺序如下；第 1–4 项均已落地（见上方状态），保留原文作为决策记录：

1. **P0：移植 Anima NAG（Normalized Attention Guidance）（已实现）**。它最直接补足 Turbo/CFG 1 下负面提示词控制，论文和 Anima 专用节点都有公开实现，MIT 许可清晰。Forge 需要增加一个可组合的 Anima cross-attention override 接口，不能直接照搬 ComfyUI 的 `optimized_attention_override`。
2. **P1 实验：Momentum Guidance（已实现）**。它针对 flow model，以跨 timestep 的速度 EMA 做外推，不增加模型前向；和 Anima 的 flow 轨迹匹配，但必须先隔离 PC3、Spectrum、SMC 的状态。
3. **P1：把 Anima 逐 block `torch.compile` 做成现有内置 Compile 插件的预设（已实现）**。这是速度优化，不改变生成算法；来源为 MIT，改动小，且比另装一个插件更符合 Forge 的无感集成方式。
4. **P2 实验：Frequency-Decoupled Guidance（FDG）（已实现，未引入 `kornia`）**。它允许低/高频使用不同 guidance 强度，适合研究低 CFG 细节，但没有 Anima 实证、占用 CFG combine，并会引入 `kornia`，不应默认开启。

其后才是 **PiD 4× 解码器**（效果潜力高但很重；已作为可选内置后处理器接入）和 **Layer Replay**（效果证据弱、原仓库缺许可证，只能 clean-room 重写）。ANIMA_BOOSTER 与 FLSampler 均为保留全部权利的代码，不能复制或移植；Anima IP-Adapter 当前仓库没有可用权重、代码不可运行，许可也只在 README 中声明，暂不应集成。

## 官方基线与使用建议

[Anima 官方模型卡](https://huggingface.co/circlestone-labs/Anima)给出的可靠基线是：

- Base/Aesthetic：512²–1536²，通常 30–50 步、CFG 4–5。
- Turbo：CFG 1、8–12 步；更稳定、更快，但多样性下降。
- 官方偏好的普通采样器包括 `er_sde`、`euler_a`、`dpmpp_2m_sde_gpu` 和 `euler`。
- 官方特别指出，RES4LYF 的 **beta57** 会更强调低噪声阶段，对写实/厚涂纹理有帮助。
- 标签提示应小写并用空格；画师标签必须有 `@`；Aesthetic 版不宜同时在正负提示中堆叠 `score_*`，否则容易把风格推得过头。

官方原生工作流可在 [Comfy Org workflow templates](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_anima_preview.json) 核对。所有新技术的 A/B 测试都应从这一基线和相同 seed 出发，而不是比较不同 prompt、不同尺寸或不同模型版本的图片。

## `remi` 已有能力

| 目标 | 当前实现 | 结论 |
|---|---|---|
| beta57 | `modules/sd_schedulers.py` / RES4LYF 移植相关代码 | 已有；官方直接推荐，不再另装 RES4LYF |
| 官方 flow 路径 | `modules/sd_samplers_anima.py`，包含 Flow Euler、UniPC2、PC3 | 已有；KeithZ117 路径已 Forge 化 |
| ER-SDE + CNS | `modules/sd_samplers_kdiffusion.py` 与 Anima guidance | 已有 |
| CFG/提示遵循 | Skimmed CFG、SMC-CFG、DCW、CNS、CLIP modulation | 已有；NAG 是新增维度，不是简单重复 |
| 预测/缓存 | `sd_forge_spectrum` 的 hidden-feature forecasting 与 SEA | 已有；TeaCache/Booster 缓存高度重复 |
| 多 LoRA 构图 | FreeFuse、artist mixer、LoRA stage/layer weight | 已有 |
| 区域构图 | `sd_forge_anima_regional` | 已有；ComfyUI-ppm AttentionCouple 基本重复 |
| ControlNet-LLLite | `sd_forge_controlllite` 的 Anima LLLite v2 | 已有；无需再移植 ComfyUI-Anima-LLLite |
| 高分辨率 | Hires Guard、MultiDiffusion、Tile preprocessor | 已有主体能力 |
| 注意力加速 | SageAttention 1/2/3 选择 | 已有 |
| 编译加速 | `sd_forge_compile` 整体模型 `torch.compile` 与 Anima per-block 预设 | 已有 |
| NegPiP | 内置 Neo 适配扩展 | 已有；无需复制 ComfyUI-ppm 的版本 |
| PiD 4× 后处理 | `backend/diffusion_engine/pid.py` + `sd_forge_pid` | 已可从当前 latent 一键进入 4-step PiD；权重需用户预先安装，与 Hires 互斥 |

## 候选技术与优先级

### P0：Anima NAG —— 最值得移植的质量功能（已实现）

[NAG 论文](https://arxiv.org/abs/2505.21179)把正负条件的外推从最终噪声预测移到 cross-attention 输出，并用 L1 范数限制外推幅度，再与原正向 attention 混合。论文报告其在少步数场景中改善负面提示控制、文本对齐和感知质量；这是作者报告的跨模型结果，**不是 Anima 专属 GPU 基准**。

[ComfyUI-Anima-NAG 固定版本](https://github.com/hybskgks28275/ComfyUI-Anima-NAG/tree/15904013437882197b8ef7dd69891e79105bf0fb)是 [MIT 许可](https://github.com/hybskgks28275/ComfyUI-Anima-NAG/blob/15904013437882197b8ef7dd69891e79105bf0fb/LICENSE)的 Anima 专用适配。其核心过程是：

1. 计算正常的正/负批次 cross-attention 输出；
2. 取正向 query 与负向 key/value 再计算一次 `z_neg`；
3. `z_tilde = z_pos + scale * (z_pos - z_neg)`；
4. 用 `tau` 对 `||z_tilde||₁ / ||z_pos||₁` 做上限约束；
5. 用 `alpha` 混合回原正向 attention。

对 Turbo 的建议起点是 `scale=2.0, tau=2.5, alpha=0.5`，仅在前 50% 步数启用。它会增加启用区间内的 cross-attention 计算，但比完整 CFG 双分支外推更有针对性。

**Forge 移植接口（实现前的状态与设计）：** 实现前 `backend/nn/anima.py::SelfCrossAttention` 直接调用全局 `attention_function`，没有 ComfyUI 的 `optimized_attention_override`；现已在 Anima attention 层加入可链式的 modifier（`backend/nn/anima_attention.py` 的 `ANIMA_ATTENTION_MODIFIERS`），只作用于 cross-attention。区域提示和 Artist Mixer 都会临时包装 `cross_attn.forward`；NAG 必须位于其下层的 Q/K/V attention 接口，才能让区域上下文分别获得正确的负向引导，且不能通过永久 monkeypatch 覆盖它们。

**必须测试的组合：** NAG × Regional、Artist Mixer、LLLite、SageAttention、Spectrum、batch > 1、Hires 二阶段、CFG 1/4.5、Turbo/Base。若没有同时包含正负分支，NAG 应明确无效果并给出提示。

### P1：Anima 逐 block `torch.compile` —— 低风险加速（排序在 Momentum Guidance 之后；已实现）

[ComfyUI-Anima-BlockCompile 固定版本](https://github.com/sorryhyun/ComfyUI-Anima-BlockCompile/tree/455eeeaeda63f1985eaa1669e4bd08cf3ec4bd13)使用 [MIT 许可](https://github.com/sorryhyun/ComfyUI-Anima-BlockCompile/blob/455eeeaeda63f1985eaa1669e4bd08cf3ec4bd13/LICENSE)，逐个编译 `diffusion_model.blocks.{i}`，而不是把整个 DiT 包成一个巨大 graph。这样可缩短首次编译、减少 graph break，并复用结构相同的 transformer block 图。

Forge 已有 `extensions-builtin/sd_forge_compile/scripts/compile.py`，因此不应新增独立扩展。建议增加 `Anima per-block` 预设：

- 仅当模型确认为 Anima 且存在非空 `diffusion_model.blocks` 时启用；
- 对每个 block 延迟 `torch.compile`，使用现有 guard filter 排除 `transformer_options`；
- 与整模型 compile 互斥，并能完整恢复原 module；
- Windows 预检 Inductor 所需 C++ 编译器，失败时回退 eager，而不是在首张图中途报错；
- 分别验证固定/动态分辨率、LoRA 切换、SageAttention 和 Hires 两阶段。

这是速度功能，不应宣称提高画质；生成结果需要做相同 seed 的数值/视觉回归，防止编译后精度路径变化。

### P1 实验：Momentum Guidance —— flow 方向匹配，但先解决状态隔离（已实现）

[Momentum Guidance 论文](https://arxiv.org/abs/2602.20360)面向 flow model，用历史速度的指数移动平均对当前 guidance 速度做外推，不增加额外模型前向；作者在 SD3、FLUX.1-dev 等 flow 模型上报告了收益。已有 [MIT 许可的 ComfyUI 实现](https://github.com/pamparamm/sd-perturbed-attention/blob/904319bff623b185f15047e231a446d615ee48c6/mg_nodes.py)，核心只依赖 PyTorch。从模型类型看它比传统 diffusion-only 技巧更值得在 Anima 上试验，但目前没有 Anima 专属基准。

它与 SMC 不同：SMC 修正 cond/uncond 控制误差，Momentum Guidance 在合并后的跨 timestep 轨迹维护 EMA；二者同时启用的数学含义未定义。Forge 原型必须在新任务、batch/shape 变化、Hires 二阶段时重置，且 PC3/UniPC 辅助前向和 Spectrum 预测/缓存都不能推进主轨迹 EMA。建议只从 Flow Euler 固定 seed A/B 开始，不默认启用。

### P2 实验：Frequency-Decoupled Guidance（FDG）（已实现）

[FDG 论文](https://arxiv.org/abs/2506.19713)把 cond-uncond guidance update 经 Laplacian pyramid 拆成低频与高频分量：低频保留结构/提示词约束，高频使用独立强度增强细节。论文报告它在低 CFG 下提高 fidelity 并减轻高 CFG 的过饱和；[MIT 节点实现](https://github.com/pamparamm/sd-perturbed-attention/blob/904319bff623b185f15047e231a446d615ee48c6/fdg_nodes.py)额外依赖 `kornia`。

它没有 Anima 实测证据，并且会完全接管 CFG combine，不能直接与 SMC、CFG-Zero* 或 APG 同时启用。它和 DCW/CNS 都涉及频域，但作用点不同：FDG 分解同一步的 guidance residual，DCW/CNS 修正跨步误差或噪声；无定义叠加仍可能过锐。上游还会把 latent padding 到下一次方尺寸，非标准长宽比会增加显存。建议将其作为 Guidance controller 的互斥实验模式，若进入 Forge 核心则自行实现小型 Gaussian/Laplacian 分解，避免把 `kornia` 变成硬依赖。**已实现**：低频段即两层 Laplacian pyramid 的 `pyrup(pyrdown(delta))`，按 kornia 的 5×5 二项核、reflect padding 与 bilinear 重采样用纯 PyTorch 重写（测试与 kornia 0.6.12 逐元素对照）；不做上游的二次幂 zero padding，奇数边长直接重采样回原尺寸。

### P2：PiD 4× 解码器 —— 高潜力、重依赖的输出后处理

[ComfyUI-Anima-PiD 固定版本](https://github.com/sorryhyun/ComfyUI-Anima-PiD/tree/a212facd634a9ee5488d5c6b6cc2f8e79e1def07)把普通 VAE Decode 替换为四步、4× 的 pixel-space diffusion 解码/超分流程。包装为 [MIT](https://github.com/sorryhyun/ComfyUI-Anima-PiD/blob/a212facd634a9ee5488d5c6b6cc2f8e79e1def07/LICENSE)，依赖 NVIDIA [Apache-2.0 PiD 实现](https://github.com/nv-tlabs/PiD)；但 [nvidia/PiD 模型卡](https://huggingface.co/nvidia/PiD#license-terms-of-use)规定权重只允许非商业研究/评估，必须单独展示许可信息。

Forge 的 `backend/diffusion_engine/pid.py` 已能把 PiD 当作独立扩散模型运行，内置 `sd_forge_pid` 也已把当前 latent 送入四步 LCM PiD 后处理并输出 4× 图像。因此不再移植第二套 vendored 网络；本轮只修复其任务级结果隔离、多批次 prompt/seed 定位、错误日志和许可/Hires GUI 说明。分块 PiD 和更细粒度的低显存管理仍属于后续能力。

优点是它直接面向最终像素细节、角落和小脸，而不是继续在 latent sampler 上叠 patch。代价是额外 checkpoint、显著显存/时间开销（上游说明的 2048 tile、bf16 路径约需 7 GB 级额外显存），并引入一条独立 decoder 生命周期。适合做“可选后处理器”，不适合默认开启或伪装成普通 VAE。

后续 GPU 验证仍需确认：模型下载与哈希、离线模式、分块 overlap、RGBA/批次、低显存卸载和取消任务。PiD 与 Hires pass 目前强制互斥：避免先生成 2K latent 再做 4× 而请求 8K 输出，超出 Qwen v1.5 2K→4K 权重的目标范围。

### P2 实验：Layer Replay —— 只能独立重写

[ComfyUI-Anima-Enhancer](https://github.com/AdamNizol/ComfyUI-Anima-Enhancer) 的做法是在后半程把指定 DiT block（默认 3/4/5）的输出再送入同一 block 一次，作者声称能增加细节与一致性。仓库的 `pyproject.toml` 指向 `LICENSE`，但当前仓库没有该文件，因此**不能复制源代码**。

这一方法没有论文或公开量化基准，而且会改变残差深度与 block 状态。若 clean-room 实现，必须作为 Experimental：明确 block 范围与 sigma 窗口；与 Spectrum、Modulation、Regional、LLLite、compile 的 patch 顺序可组合；用至少 20 seeds 比较结构错误率、提示遵循和纹理，而不能只挑选样图。

### P2/P3：Guidance Limiter —— 有论文依据，但需先证明适合 Anima Base

[ComfyUI-ppm 的固定版本 Guidance Limiter 源码](https://github.com/pamparamm/ComfyUI-ppm/blob/00ede55ac1d58e750b5c456245ca463597e08557/src/nodes_ppm/guidance.py)基于论文 [Applying Guidance in a Limited Interval](https://arxiv.org/abs/2404.07724)，在早期/晚期区间关闭 CFG，只在中间区间保留引导。仓库为 AGPL-3.0；原作者的[参考实现](https://github.com/kynkaat/guidance-interval)为 Apache-2.0。

它对普通 CFG 模型有明确研究依据，但 Turbo 本来就是 CFG 1，因此对 Turbo 没意义；对 Anima Base/Aesthetic 可能降低过饱和并改善分布质量。PPM 默认 `sigma_start=5.42/sigma_end=0.28` 来自非 flow schedule，不能用于 Anima 的 `0..1` flow sigma；Forge 应存储归一化采样进度，再映射到实际 schedule。当前 Guidance 扩展已有多种 CFG 控制，最佳集成方式是增加一个“CFG active range”，而不是预装整个 PPM 节点包。**已实现**为 Anima Guidance 的 CFG active range（界面 “Limit CFG to a sampling range”）；起止值按 Forge 的 `percent_to_sigma` 解释为模型去噪时间比例（与 Forge 自身的 start/end percent 相同），不随 scheduler 变化，因此覆盖的步数比例取决于调度器；仍需以 Base/Aesthetic、CFG 4–7 做 A/B。

PPM 的 AttentionCouple 和 NegPiP 已分别被当前 Regional 与 NegPiP 覆盖；AttentionCouple 会再次占用 attention/mask patch seam，与 Regional 同时启用还会产生未定义的 hook 顺序、mask resize 和 Spectrum cache 语义，因此不移植。动态采样器和 CFG++ 也不能未经 flow/const 数学验证直接套到 Anima。

### P2 实验：SPEED（Spectral Progressive Diffusion）—— 训练无关的前段半分辨率加速（已实现）

[SPEED 论文](https://arxiv.org/abs/2605.18736)（Stanford，2026-05）利用 latent 功率谱的幂律衰减：高频在去噪早期仍被噪声主导，因此前段可在低分辨率上运行；到 δ-optimal 切换时刻（Eq. 9–10，由功率谱拟合 `P(ω)=A·|ω|^-β` 和单一容差 δ 决定）在 DCT 域把低分辨率谱嵌入全分辨率网格，新暴露的高频系数按当前噪声水平 `t·ε` 填充，再按 Eq. 5–6 缩放状态并把时间对齐到 `t̃ = r·t / (1 + (r−1)·t)`。论文报告 FLUX.1-dev 训练无关 S=2、δ=0.01、DCT 时 1.66× 墙钟加速（ImageReward 1.049 对 50 步基线 1.095），与减步结合最高约 7×；WAN 2.1 视频约 2×。[官方代码](https://github.com/howardhx/speed)与[在 Anima 上测试的 ComfyUI 移植](https://github.com/ruwwww/ComfyUI-SPEED)均为 MIT（移植 README 把官方写成 BSD-3，但官方 LICENSE 文件是 MIT）；移植作者用 FLUX 功率谱预设在 Anima 上报告 1.33×，没有质量指标。

Forge 实现为 `Anima SPEED` 采样器（`modules/sd_samplers_anima_speed.py`），固定论文默认值：两阶段 0.5→1、δ=0.01、DCT、Euler、锁定 Anima FlowMatch 调度。功率谱采用论文 Table 4 的 Wan 2.1 VAE latent 拟合（A=219.48，β=2.4227），因为 Anima 使用 Wan VAE；1024² 时切换 sigma≈0.694，30 步中前 18 步在 64×64 latent 上运行，832×1216 时前 16 步。与参考实现的差异：初始低分辨率噪声是 seed 噪声的 DCT 截断，切换时填充的高频直接复用被截掉的 seed 噪声系数（与低频轨迹独立，统计上等价于新抽样，且结果随 Forge seed 可复现、更接近同 seed 的全分辨率结果）；缩放比按实际网格 `sqrt(HW/hw)` 计算，奇数 latent 尺寸时与名义 2 略有差别。

**Anima 实测功率谱与切换点（2026-10 评估）**：[anima_lora `703b34b` 的 `docs/inference/spd.md`](https://github.com/sorryhyun/anima_lora/blob/703b34b40e3b724b20bf1a006168023241c59584/docs/inference/spd.md)（[MIT](https://github.com/sorryhyun/anima_lora/blob/703b34b40e3b724b20bf1a006168023241c59584/LICENSE)，同一提交已用于下文 DAVE、xattn boost、FSG、T-GATE 条目）在 Anima latent 上测得 `P ∝ |ω|^-β`，β=2.26（R²=0.9994，200 张图，30/30 个画师落在 [2, 3]）；其 Phase 1 测量显示 δ=0.01 的原则调度在 Anima 上只允许约 ×1.15–1.22；832×1216、2 seed、墙钟的调度对比中，单次切换 σ0.7 为 ×1.32，σ0.5 为 ×1.65（作者推荐的拐点），0.5/0.75/1.0 两次切换 σ0.7/0.5 为 ×1.61，最激进的 0.4/0.7/1.0 为 ×1.73；作者同时写明 SPD 在 Anima 上的输出稳定地更锐、对比更高，“不是透明的加速”，没有量化画质指标；其 runner 使用手设切换 sigma，不用 Eq. 9–10。**证据等级：β 拟合本身扎实，调度对比是作者 2 seed 的目测 bench。**决定：（1）**不采用 β=2.26**。Eq. 10 的切换点取决于联合拟合的 `A·ω^-β`，spd.md 只给出 β，没有给出幅度 A 和频率/功率归一化，它引用的测量脚本（`_archive/spd/bench/`）也不在该提交的仓库中；保留论文 A=219.48 只换 β 会让 ω=32（1024² 的半分辨率上限）处的功率变为约 1.76 倍，这是两份拟合的混合，并非 Anima 实测。若照此代入，1024² 的切换 sigma 由 0.694 升到约 0.754（30 步中半分辨率步 18→15），832×1216 由 0.748 升到约 0.799（16→13），更保守、加速更少，而没有证据表明当前切换点损失画质——唯一的 Anima 画质证据（上面的 bench）反而认为更晚切换仍可接受。当前默认在 832×1216 于第 16 步（σ≈0.72）切换，大致对应作者的单次 σ0.7 设置。（2）**采用有界的切换 sigma 覆盖**：设置 → Sampler Parameters 的 `[Anima SPEED] Switch sigma`（`anima_speed_switch_sigma`），默认 0 = δ-optimal；非 0 值限制在 [0.5, 0.9]：作者在 Anima 上 bench 过的切换点是 0.5–0.8（0.5 是最晚的单次切换；更低没有 Anima 证据，且作者测得最高频带到 σ≈0.29 才定形），0.9 是没有 bench 的保守延伸。自动值只取决于短边：短边 392 px 起不高于 0.9（只有短边 376–384 px 的最窄长条略高，最多 0.903），短边 1984 px 起才低于 0.5（2048² 为 0.49），此时 0.5 的覆盖反而比自动更早切回全分辨率，从第一个 sigma ≤ 该值的步开始全分辨率；非 0 时写入 infotext `SPEED switch sigma`，粘贴时作为设置覆盖恢复（与 `Eta` 相同的机制），img2img/Hires 等回退到 Flow Euler 的遍也照样记录。这让用户按固定 seed 对比自行在速度与贴近全分辨率结果之间取舍，默认行为不变。如果之后得到 Anima latent 上与论文代码同一归一化的 A 与 β（公开的拟合或本地用论文脚本测量），再重新评估默认值。

它只在 txt2img 从纯噪声开始时生效：img2img/denoise<1/Hires 二阶段、inpaint mask、Regional/Artist Mixer/FreeFuse、任何绑定全分辨率网格的 `model_function_wrapper`（Spectrum、MultiDiffusion、Radial；标记为 `__resolution_independent__` 的 Modulation、CFG-Zero* zero-init、DAVE 与 Cross-Attention Boost 不触发回退）、ControlNet、LLLite/IP-Adapter patch、参考 latent，以及半分辨率小于 512×384 或调度没有可切换步骤时，都退回 Anima Flow Euler 并写入 `Anima conflict resolution` infotext。Momentum 因不在 Momentum 采样器集合中被自动关闭；SMC/APG/DCW/CFG-Zero* 的跨步状态在切换分辨率时重置。前段预览为低分辨率；前段被中断时返回双线性放大到全尺寸的 latent；per-block compile 会为第二种形状重新编译。CPU 测试覆盖 DCT 正交性、低频保持、注入噪声方差等于对齐 sigma、阶段单调并以全分辨率结束、输出形状、闭式高斯 RF 去噪器下与同 seed 全分辨率 Flow Euler 一致（去掉对齐则偏离）、1024² 与 832×1216 的默认切换 sigma 与步数、切换 sigma 覆盖的边界与 infotext，以及各类回退；Remi 自身尚未在 GPU 上测量 Anima 的画质与实际加速，所用功率谱拟合来自 Wan VAE 而非 Anima 图像 latent（Anima 上的 β 见上段）。

### P2 实验：DAVE（DC Attenuation for diVersity Enhancement）—— 前段 DC 衰减提升同提示词多样性（已实现）

[DAVE 论文](https://arxiv.org/abs/2606.06813)（ICML 2026，[官方代码](https://github.com/daheekwon/DAVE) MIT，SD3 diffusers 改版）发现同一 prompt 不同 seed 的中间 Transformer 特征的空间均值（DC 分量）在去噪早期迅速趋同，把轨迹锁进相似构图；DAVE 在前 `tau` 步把选定 block 输出的 DC 按比例衰减 `h ← h − s·mean(h)`，保留 seed 特有的 AC 残差，训练无关、几乎无额外开销。[ComfyUI-Anima-DAVE 固定版本](https://github.com/sorryhyun/ComfyUI-Anima-DAVE/tree/83143e8d84768e25f72755ec00ea00ded07ee06e)为 [MIT](https://github.com/sorryhyun/ComfyUI-Anima-DAVE/blob/83143e8d84768e25f72755ec00ea00ded07ee06e/LICENSE) 的 Anima 专用移植：其 `dave_alpha.npz` 由作者在 Anima 2B 上逐 block 测量早期 DC 跨 seed 相似度、功率比与 DC–AC 差距得出，取平坦的 block 8–18；默认 `strength=0.30`、`tau=0.10`，并说明窗口宽度比强度更伤文字/手（`tau≥0.15` 明显劣化），多样性常在低强度（0.05–0.2）达到峰值。**证据等级：Anima 上仅有作者 seed 网格的定性结果**，没有 FID/Vendi 等量化指标。

Forge 实现为内置扩展 `sd_forge_anima_dave`（默认关闭、仅 Anima）：`model_function_wrapper` 按 `sampling_sigmas` 的完整调度判断窗口（与参考相同的 `step < round(tau·n)`，至少 1 步），只在窗口内的调用中用 `forward_overrides` 临时替换 block `forward` 并在 `finally` 中恢复；均值按样本、按通道只在生成图像的 token 帧上计算，edit 模型追加的参考 latent 帧不参与也不修改。block 列表按 2B 布局书写，在 40/52 层模型上映射到原始 block（不含插入副本），保持被衰减的 block 数量与参考一致。与参考的差异：窗口基于完整调度，因此 Hires/img2img 只有 denoise 覆盖窗口时才生效（参考在 ComfyUI 中按截断后的调度计算）；调度缺失时不生效（参考会每步生效）；`tau` 滑块最小 0.01，不提供参考中 `tau=0` 表示每步的语义。Spectrum 在窗口内被强制实际前向，per-block compile 在窗口内走 eager；token 均值与分辨率无关，wrapper 标记 `__resolution_independent__`，SPEED 保留半分辨率前段；Regional/FreeFuse/Artist Mixer 作用于 cross-attention，可组合；PC3 探测按其自身 sigma 判断（无状态，无需特判）。CPU 测试覆盖 DC 衰减数学（均值按比例缩小、AC 不变、dtype）、参考帧隔离、窗口门控、28/40/52 block 映射、异常时恢复 forward、wrapper 链与 Spectrum/SPEED 标记、infotext 键；GPU 上的多样性与画质尚未验证。

### P2 实验：Front-loaded Cross-Attention Boost —— 高噪声前段放大正向 cross-attention 改善提示词遵循（已实现）

[anima_lora `703b34b`](https://github.com/sorryhyun/anima_lora/blob/703b34b40e3b724b20bf1a006168023241c59584/docs/inference/xattn_boost.md)（[MIT](https://github.com/sorryhyun/anima_lora/blob/703b34b40e3b724b20bf1a006168023241c59584/LICENSE)，作者同 ComfyUI-Anima-DAVE）的 `--xattn_boost`：作者探针显示 Anima 的 cross-attention 文本驱动只在 σ≈1 附近存在（σ<0.85 后约 0.02），因此在 σ ≥ 0.85 的“构图写入”窗口内把每个 block 的 cross-attention 残差（AdaLN gate 之后）乘以 λ，只作用于条件前向，增益全部进入 CFG 差值；随后按图像把 token 范数均值拉回未增强时的值（`img` 模式，`scale**ρ`，ρ=0.5），使增强成为朝文本方向的旋转而不是离开训练范数壳的残差叠加。1× NFE，不增加前向。**证据等级：Anima 上仅有作者目测评审的 bench**（`bench/frontload_text_boost/report.md`，2B base，17–20 个手选 prompt，1–3 seed，28 步 er_sde、flow shift 3）：λ=2 修复 7 个基线失败标签中的 5 个，同窗口提高 CFG 最多修复 2 个且风格崩坏；副作用是放大所有标签（`cropped`/画师取景习惯、人数、身份细节漂移），画师/风格标签会稀释内容收益，未知概念无效；ρ 与范数匹配模式由目测选定，λ∈{2.5,3} 与 0.85 以外的频带未测，没有量化指标。与 DAVE 接受时的证据等级相同，且上文 FSG 拒绝条目中作者正是以它作为胜过 FSG 的 1× NFE 手段。

Forge 实现为内置扩展 `sd_forge_anima_xattn_boost`（默认关闭、仅 Anima；强度 2.0、频带 0.85、范数匹配 0.5）：`model_function_wrapper` 按每次模型调用的 sigma 判断频带，窗口内用 `forward_overrides` 临时替换各 block 的 `cross_attn.forward`，按 `cond_or_uncond` 标记（`branch_rows`）只放大正向行，CFG 1 时全部为正向行；范数匹配另外临时包装 `self_attn` 与 block `forward`，由捕获的 self/cross-attention 输出与重算的 AdaLN gate 重建 cross-attention 前后的残差，按参考公式求每图缩放 `s`；参考在 MLP 之前缩放 `x_new`，而 MLP 经过无仿射 LayerNorm 读取它（对 `s` 不变，仅差 eps），因此等价地把 `(s−1)·x_new` 加到 block 输出。CPU 测试用真实 `Block` 与按参考 `Block._forward` 重写的实现逐元素对照（ρ=0/0.5/1，1e-5 以内）。与参考的差异：参考按采样步的 σ 切换，这里按每次调用的 σ（单次评估采样器相同；PC3 探测按自身 sigma 判断，无状态，无需特判）；频带是绝对 sigma，Hires/img2img 只有 denoise 覆盖频带时才生效，被增强的步数随调度器与 shift 变化；2.9B/3.8B 上所有 block 都增强（参考只测过 2B）；edit 模型的参考 latent 帧同样参与。Spectrum 在频带内被强制实际前向，per-block compile 在频带内走 eager；增益与每图范数与分辨率无关，wrapper 标记 `__resolution_independent__`，SPEED 保留半分辨率前段；NegPiP、NAG、Regional、FreeFuse、Artist Mixer 都作用在 cross-attention 内部，增强缩放其输出，可组合；批次与标记不符的嵌套调用保持不变。CPU 测试另覆盖 λ=1 恒等、负向行逐位不变、范数匹配保持每图 token 范数均值、频带门控、异常时恢复 forward、wrapper 链与 Spectrum/SPEED 标记、infotext 键；GPU 上的遵循度收益与副作用尚未验证。

### P3：Dynamic CFG / 频域细节 —— 只适合作为研究原型

[ComfyUI-AnimaDynamicCFG 固定版本源码](https://github.com/DanrisiUA/ComfyUI-AnimaDynamicCFG/blob/35b70e0c402ea52e60aa18aafd87cce6ba3b0c55/nodes.py)包含 CFG 曲线、自适应 CFG、CFG rescale、latent 均值校正、FFT 高频增强和后期随机噪声，README 现声明 **MIT**（2026-09 核对 `35b70e0`），但仓库仍没有 LICENSE 文件，也就缺少 MIT 要求随代码保留的版权与许可声明；许可已不是主要障碍。不移植的理由在于它没有测试和量化证据，当前实现还存在几项方法学问题：

- 自适应 CFG 使用 `cond-uncond` 的**绝对整体 L2 norm**，随分辨率和 batch 改变，默认 `target_divergence=1` 没有跨尺寸意义；
- 进度依赖 `sample_sigmas`，缺失时退化成固定 1→0，未证明符合 Anima 的实际 sigma；
- 高频放大和随机噪声可能只是制造锐化/颗粒感，也可能破坏 flow 轨迹；随机噪声使用全局 `torch.randn_like`，没有绑定 sampler generator，固定 seed 也不保证可复现；
- “更写实”与官方将 Anima 定义为非写实模型的目标并不一致。

其中标准 CFG Rescale 已有论文依据，但其余功能不应直接移植；scheduled CFG 与频域修正分别与现有 Stage Scheduler、DCW/CNS 部分重叠；CFG rescale 针对的高 CFG 过饱和由已实现的 APG 处理（APG 论文以 CFG rescale 为对照），不再单独移植（`sd-forge-anima-prompt-rescale` 缩放的是 `[from:to:when]` 的步数，与 CFG 无关）。若研究，应依据公开算法 clean-room 重写，先把 divergence 改成每元素 RMS/robust norm，并加入确定性 seed 与 flow-aware 进度。

### P3：Anima IP-Adapter —— 方向重要，当前仓库不可交付

[comfyui-anima-ipadapter](https://github.com/Wenaka2004/comfyui-anima-ipadapter) 宣称以 SigLIP2/InstantCharacter 风格的 decoupled cross-attention实现参考图人物一致性。这是 Anima 生态真正缺少的重要方向，但当前仓库不能移植：

- 许可只在 README 中声明为 **Apache-2.0**（仓库仍没有 LICENSE 文件）；README 描述的是以 [Qwen3-VL-Embedding-2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) 1024 维图像嵌入驱动 Perceiver Resampler 的旧版，而最新提交（`5f53e2b`，2026-05-15）已改为 SigLIP + InstantCharacter 式 TimeResampler，文档与代码不一致；
- README 没有可下载的训练完成权重，[issue #2](https://github.com/Wenaka2004/comfyui-anima-ipadapter/issues/2)至 2026-09 仍开放，只有用户追问 Anima checkpoint 而没有作者发布；评论中提到的第三方 [LuciferTC/Anima-IP-Adapter](https://huggingface.co/LuciferTC/Anima-IP-Adapter) 未声明许可，且据报告无法运行；
- `5f53e2b` 中 `TransformerEncoderLayer` 仍被以不支持的 `memory=` 参数调用（`ip_adapter_model.py:66`），loader 仍向 `IPAdapterSigLIP` 传入不存在的 `input_dim=` 参数（`nodes.py:116`）；
- cross-attention wrapper 的函数签名与当前 Forge/Comfy Anima 路径不一致；
- patch 生命周期、逐步 timestep 更新和卸载逻辑不完整。

许可声明已补（仅 README），但拒绝理由仍然成立：应等待上游发布权重和可运行版本后再评估。不能把训练脚本或空壳节点预装给用户。

## 已有或不应移植的项目

### 已经覆盖

- [ComfyUI-Anima-LLLite](https://github.com/kohya-ss/ComfyUI-Anima-LLLite)：Apache-2.0，但当前 Forge 已内置更完整的 Anima LLLite v2，包括图像/掩码、时间范围和多控制组合。
- [TeaCache](https://github.com/welltop-cn/ComfyUI-TeaCache) / [TeaCache 论文](https://arxiv.org/abs/2411.19108)：属于 timestep-aware 缓存；当前 Spectrum 已做 Anima 专用预测与 SEA。可以借鉴评测指标，但再叠缓存会造成双重跳步和状态冲突。同族的后续方法（MagCache、EasyCache、First-Block-Cache、TaylorSeer、FoCa）同样是按步复用或外推整个前向/残差，与 Spectrum 的逐步特征预测占据同一接缝，结论相同（2026-09 第十七轮评估）：只作为 Spectrum 调度与 SEA 的评测对照，不作为第二套缓存移植。
- [SageAttention](https://github.com/thu-ml/SageAttention)：Forge 已内置多版本选择；官方实现也提醒应在目标 DiT attention 类内替换，而不是全局替换 PyTorch SDPA。
- [ComfyUI-vslinx-nodes](https://github.com/vslinx/ComfyUI-vslinx-nodes) 的 Anima LLLite tiled sampler：当前已有 MultiDiffusion + Anima LLLite 主体能力。只有在确认控制图 crop 与 latent tile 完全同步方面存在缺口时，才值得抽取该逻辑。
- Anima style browser：属于提示词 UI；现有 TagComplete、Prompt-All-In-One 和 Artist Mixer 已覆盖更适合 Forge 的入口。

### 许可禁止直接移植

- [ANIMA_BOOSTER 固定版本](https://github.com/BlackSnowSkill/ANIMA_BOOSTER/tree/0cb1f45f4c6b8a726150faf9c08988bc10baf36c)：没有 LICENSE，README 声明保留全部权利并禁止未经许可复制、修改、合并、分发或托管。它的 SageAttention、compile、缓存和 latent 对齐大部分已被当前 Forge 覆盖；唯一值得独立实现的逐 block compile 已有 MIT 来源。
- [ComfyUI-BSS_FLSampler 固定版本](https://github.com/BlackSnowSkill/ComfyUI-BSS_FLSampler/tree/3c4ca98ba4f3137c8234a963ddce34b44cfa6035)：同样没有 LICENSE，README 声明保留全部权利并禁止复制、修改、合并与托管。其“相邻 x0 差分→池化阈值掩码→latent unsharp mask→衰减随机纹理”的算法只有作者描述，缺少论文和独立基准。不能复制源码；在取得许可前也不建议优先 clean-room 复现。

### 不适用于标准 Anima checkpoint

- [Anima Native Reference](https://github.com/HK416-TYPED/ComfyUI-Anima-Native-Reference) 需要带专用 router/adapter 的集成 checkpoint，不是普通 Anima 的无权重生成优化。

## 中长期论文候选

这些论文有研究价值，但没有证据证明可直接套用当前 Anima，暂不进入移植队列：

- [CFG-Zero*](https://arxiv.org/abs/2503.18886)：针对 flow matching 的早期 zero-init 和优化 guidance scale；在 Flux/SD3 等模型上有结果。**已实现**：依论文 Algorithm 1 独立编写（未复制无许可证的作者仓库），作为 `CFG-Zero*` CFG 模式加入 Anima Guidance。s* = ⟨v_cond, v_uncond⟩ / ‖v_uncond‖² 逐样本计算；由于 Anima 为 rectified flow，Forge CFG hook 的残差 `x - denoised` 等于 `sigma * v`，而 s* 对同一正比例因子不变，故直接在残差上组合（ComfyUI `CFGZeroStar` 是在 x0 空间的 post-CFG 近似，两者不完全等价）。zero-init 让从纯噪声开始的 pass 的前 N 个不同 sigma 的采样步输出零速度（`denoised = x`，latent 不变），PC3 辅助探测不计步；Hires/img2img pass 从调度中段开始，不做 zero-init。仍待 GPU 固定 seed 矩阵验证 Anima 上的收益。
- [Adaptive Projected Guidance（APG）](https://arxiv.org/abs/2410.02416)：把 CFG 更新分解为平行/正交于条件去噪预测的分量，减弱平行分量以避免高 CFG 过饱和，并加入范数上限与反向 momentum。**已实现**为 `APG` CFG 模式：论文强调投影必须作用在去噪预测上才有效，因此直接使用 Forge 的 `cond_denoised`/`uncond_denoised`（Anima rectified flow 的 x0 预测），而不是 `sigma * v` 残差；组合式为论文的 `cond + (w − 1)·(Δ⊥ + η·Δ∥)`，`eta = 1`、无上限、无 momentum 时等于普通 CFG。CFG 1 时直接返回条件预测，不让 Forge 的零占位 uncond 进入 momentum。norm threshold 是整张 latent 的绝对 L2 范数，随分辨率变化，默认关闭；论文 SDXL 设置为 `eta 0`、`r 15`、`β −0.5`。仍待 GPU 固定 seed 矩阵验证。
- [TCFG: Tangential Damping Classifier-free Guidance](https://arxiv.org/abs/2503.18137)（CVPR 2025，论文 CC BY 4.0）：**已实现（2026-09-30 评估）**为 `TCFG` CFG 模式。每步把无条件与条件预测拼成 2×N 矩阵 `A = [s_u; s_c]`，将无条件预测投影到 `A` 的第一右奇异向量 `v1` 上、丢弃切向分量，再按普通 CFG 组合。证据等级：论文实验（非 Anima）——MS-COCO 30k 零样本 FID：SD3（rectified flow）16.66 → 13.74、SDXL 13.36 → 12.65、SD1.5 13.26 → 13.12，CLIPScore 均不变；ImageNet DiT FID 32.67 → 29.5；论文称不增加推理时间。作者项目页的代码仍为 “coming soon”，没有官方仓库；参考实现为 [ComfyUI 内置 `TCFG` 节点 `a061b06`](https://github.com/Comfy-Org/ComfyUI/blob/a061b06321b4e91d05c7c436b1e9b188360c5377/comfy_extras/nodes_tcfg.py)（GPL-3.0，首次加入于 `f02de13`），它在 pre-CFG hook 中对 `x - denoised` 做逐样本 `torch.linalg.svd` 后替换 uncond。依论文 clean-room 编写，只借参考节点确认作用空间：Forge 的残差 `x - denoised` 对 Anima 等于 `sigma * v`，`v1` 对同一正比例因子不变，所以在残差（速度）上计算、不在 x0 上（与 APG 不同）。为避免 N 维 SVD 与主机同步，`v1` 取自 2×2 Gram 矩阵的闭式主特征向量，投影化简为 `e0·(e0·u + e1·c)`，其中 `|u|² − |c|²` 以 `(u − c)·(u + c)` 计算；在 1024² 尺寸、cond/uncond 高度相关时，其 guidance 方向相对 float64 SVD 的误差（≈1e-5）小于 float32 SVD（≈1e-3）。float32 计算、逐样本、`w = 1` 时精确返回 cond，CFG 1 零占位时直接返回条件预测；两奇异值相等（正交且等范数）时保留 uncond（普通 CFG）。测试覆盖与 `torch.linalg.svd` 投影的一致性、batch 独立性、半精度、残差/速度等价、AND 权重与 CFG 1 占位规则及 infotext/preset 注册。仍待 GPU 固定 seed 矩阵验证 Anima 上的收益。
- [On the Guidance of Flow Matching](https://arxiv.org/abs/2502.02150)：给出一般 flow guidance 框架，更适合作为校验现有/未来引导算法的理论参考。
- [Token Perturbation Guidance（TPG）论文](https://arxiv.org/abs/2506.10036)及其 [MIT 官方实现](https://github.com/TaatiTeam/Token-Perturbation-Guidance)：对中间 token 做保持范数的 shuffle，以原始/扰动预测之差产生 guidance。官方只验证 SDXL、SD2.1；Anima 需要选定 block、增加额外预测，并会与 Spectrum、FreeFuse、Modulation 共用 block/attention seam，只适合开发者研究开关。
- [Perturbed Attention Guidance（PAG）论文](https://arxiv.org/abs/2403.17377)及 [Diffusers 官方管线](https://huggingface.co/docs/diffusers/api/pipelines/pag)：理论上可作用于 Anima self-attention，但没有 Anima 实证，会增加预测成本，并与 FreeFuse/Spectrum 的 attention/cache 路径高度敏感，优先级低于 NAG、MG、FDG。
- Skip Layer Guidance（SLG）与其论文形式 [STG（Spatiotemporal Skip Guidance，arXiv:2411.18664）](https://arxiv.org/abs/2411.18664)：**deferred（2026-09 第二十六轮评估）**。在 sigma 窗口内对条件分支再跑一次跳过选定 block 的前向，把 `scale·(cond − cond_skip)` 加到 CFG 结果上（[ComfyUI `SkipLayerGuidanceDiT` `4d172e9`](https://github.com/Comfy-Org/ComfyUI/blob/4d172e9ad7c50d08f21df48b04cca9b5f551d0e7/comfy_extras/nodes_slg.py)，GPL-3.0，与本仓库 AGPL-3.0 兼容；最初是 Stability AI 为 SD3.5 做的实验实现，另有只改 uncond 分支的 Simple 变体）。推迟理由与上面的 PAG 相同且更具体：（1）没有任何 Anima 证据，STG 论文只评测视频 DiT，社区用例集中在 SD3.5/Wan；（2）跳哪些 block 与模型强相关，默认 `7, 8, 9` 是为 SD3/Flux 的双流/单流结构选的，Anima 的 28 个同构 block 没有对应的测量，而 anima_lora 的测量显示 self-attention 与 MLP 在每个 σ 都主导残差更新（`crossattn_self_attn_dominance.md` Result 2），整块跳过的扰动幅度与 PAG 只替换 attention 图不同，需要实验确定；（3）每个窗口内的步骤多一次单分支前向，Spectrum 需要强制实际前向并多一个分支键，FreeFuse/Regional/Artist Mixer/DAVE 在被跳过的 block 上的 patch 与 per-block compile 的编译图都要定义跳过时的行为，还要与 NAG/APG/TCFG 等 CFG 模式定义叠加次序。**实施条件**：Anima Base 上固定 seed 对照（≥20 seeds），与同 NFE 的对照组（增加步数或同等额外前向）相比在结构错误（手、肢体）上有稳定收益，并测出合适的 block 范围；届时作为默认关闭的 Guidance 选项实现，block 字段沿用 28 层布局。本轮不写代码。
- [Adaptive Guidance（AG，arXiv:2312.12487）](https://arxiv.org/abs/2312.12487)：**rejected（2026-10 第二十八轮评估）**。当条件与无条件预测的余弦相似度超过阈值后，余下步数不再跑无条件分支（论文称画质不变、计算减少约 25%，另有用历史 score 的仿射组合代替部分前向、会偏离基线的 LinearAG）；只有论文，clean-room 实现无许可障碍，也可以接在现有 CFG active range 的 uncond 跳过 hook（`make_guidance_range_uncond_skip`）上。拒绝理由是停止判据不适合 Anima：[anima_lora `crossattn_self_attn_dominance.md`](https://github.com/sorryhyun/anima_lora/blob/703b34b40e3b724b20bf1a006168023241c59584/docs/findings/crossattn_self_attn_dominance.md) 的测量显示 σ<0.85 后 `cos(v_cond, v_uncond)` 已约 0.997（方向几乎一致，文本只做缩放），但 `‖v_cond − v_uncond‖/‖v_cond‖` 在 σ≈0.66 仍约 0.035，CFG 按 `(w − 1)` 放大这一差值（CFG 4.5 时约为速度的 12%），并且 Result 5 表明需要画出文字等细小特征时这一差值在 σ 0.8–0.45 还会升高；只看方向的判据会在 σ≈0.85 附近就关掉 CFG，丢掉仍然有效的缩放引导。若要按幅度自适应判断，就是没有依据的新方法。固定区间的节省已由 CFG active range（Limited Interval，见上文 Guidance Limiter）提供，且可测试、可复现。只有在 flow 模型上出现同 NFE 对照、证明某种自适应判据优于固定区间时再评估。
- anima_lora 的 flow 形式 CFG++ 重加权（[`fsg_core.py` `cfgpp_guidance_weight` @ `ebc474e`](https://github.com/sorryhyun/anima_lora/blob/ebc474e5bb7b1880f58fde47de69b4b51c313235/library/inference/corrections/fsg_core.py)，MIT，`--cfgpp`，默认 λ=1.5）：**deferred（2026-10 第三十三轮评估）**。作者把 CFG++ 在 flow 模型上推导成对 cond/uncond 组合的纯重加权 `w_eff = λ·(1 − σ')·σ/(σ − σ')`，可接入任何积分器，与 TCFG 一样有界、可在 CPU 上测试，上文 FSG 条目中“`remi` 没有经过 flow 数学验证的 CFG++”的阻碍因此不再是推导本身。推迟理由：（1）唯一的 Anima 证据是作者的 λ 扫描（“最接近 CFG=4”），即与调好的普通 CFG 画质相当，没有显示优于 CFG 的对照；（2）`w_eff` 与步长 `σ − σ'` 成反比：28 步 shift 3 时中段约 11，步数加倍到 56 步时约翻倍，引导强度随步数变化，这与用户对 CFG 值不随步数改变的预期相反，会让现有预设与 CFG 滑块的含义失效；（3）Forge 的 CFG hook 按每次模型评估调用，Heun、PC3 探测、UniPC 校正等多次评估采样器上“下一个 σ”没有唯一定义（与 FSG 条目的问题相同）。**实施条件**：Anima 上与调好的 CFG 同 NFE 对照显示稳定收益，并为多次评估采样器定义 `σ'`；届时作为默认关闭的 CFG 模式实现。
- CJK 词表包（[anima_lora `docs/methods/cjk_vocab_pack.md` @ `ebc474e`](https://github.com/sorryhyun/anima_lora/blob/ebc474e5bb7b1880f58fde47de69b4b51c313235/docs/methods/cjk_vocab_pack.md)，权重在 [HF `sorryhyun/anima-vocab-pack-cjk`](https://huggingface.co/sorryhyun/anima-vocab-pack-cjk)）：**deferred（2026-10 第三十三轮评估）**。为日/韩/中文提示词在 T5 侧增加额外 embedding 行（id ≥ 32128），通过分词重路由和 `llm_adapter.embed` hook 接入；不含 CJK 字符的提示词分词结果逐位不变。与 `remi` 的中文用户相关，但推迟：（1）证据只有同 seed 的定性网格，且训练行主要针对画面中的日文文字（preview4 的 1,362 个训练行为假名、汉字和标点，用于画出引号内的日文），没有中文提示词的对照；（2）仍是 preview，几周内从 preview 更新到 preview4，行表与 sidecar 格式随版本变化；（3）权重许可为 circlestone-labs 非商业许可（与 Anima 模型本身相同，用户已受其约束，但它是需另行下载的权重资产）；（4）Forge 侧需要在 Anima 文本引擎中加入分词重路由与 embed hook，并让条件缓存键、NegPiP 的 token 权重和 FreeFuse 的 token 位置识别这些新行。**实施条件**：出现非 preview 的稳定版本，并有中文提示词的同 seed 对照；届时作为默认关闭、可选下载的文本编码器资产接入。
- [Optimal Stepsize for Diffusion Sampling（OSS，arXiv:2503.21774）](https://arxiv.org/abs/2503.21774)：**deferred（2026-10 第三十六轮评估）**。用动态规划从多步参考轨迹“蒸馏”出给定步数下误差最小的 sigma 表，论文称在 flow 模型上 10 步保留约 99.4% 的 GenEval。[官方代码 `ee35043`](https://github.com/bebebe666/OptimalSteps/tree/ee350436c86e29088c0cf550300a2e6a5911a7ee) 为 MIT；[ComfyUI `OptimalStepsScheduler` `174208d`](https://github.com/Comfy-Org/ComfyUI/blob/174208df6ba033e5b4784c94d83d065c0d4165a4/comfy_extras/nodes_optimalsteps.py) 只内置 FLUX、Wan、Chroma 三张预先搜出的 sigma 表，按对数线性插值到任意步数。推迟理由：（1）没有 Anima 的表：搜表需要用 Anima 本身跑多步参考轨迹做校准，属于要 GPU 的离线计算，本仓库无法产出，也没有公开的 Anima 表；借用 FLUX/Wan 的表没有依据（Anima 的 shift 与训练噪声分布不同）；（2）Base 的默认调度（Beta57、Anima FlowMatch）与 Turbo 的蒸馏步数已有各自的证据，OSS 表是否优于它们需要同 NFE 对照；（3）搜出的表绑定采样器与 shift，在 Forge 中还要定义与 Anima Flow 采样器调度锁、Hires/img2img 起始 sigma 的关系。插值本身有界、可在 CPU 上测试。**实施条件**：用官方代码在 Anima Base 上搜出 10–16 步的表，并在固定 seed（≥20 seeds）下与同 NFE 的 Beta57 对照显示稳定收益；届时作为带固定表与对数线性插值的调度器（如 “Anima OSS”）加入，并补 CPU 插值测试。本轮不写代码。
- [Rectified-CFG++](https://rectified-cfgpp.github.io/)是 flow 模型的 predictor-corrector guidance；其求解路径与已经移植的 KeithZ117 Flow Corrective / PC3 高度重叠。除非同一基准证明它有独立收益，否则不再加入第二套 predictor-corrector sampler。
- [Foresight Guidance（FSG，NeurIPS 2025）](https://arxiv.org/abs/2510.21512)：**rejected（2026-09 第六轮评估）**。在抽样步前把 `x_t` 用 K 次“CFG 引导去噪 Δσ → 无条件反演回 σ”的不动点迭代校准到 cond/uncond 一致的 golden path，每个启用步额外 `3·K` 次单分支前向。许可无障碍：[官方代码 `012398f`](https://github.com/Ka1b0/Foresight-Guidance/tree/012398fae56912f88fd8fec588b4ceb92800d9d6)、[ComfyUI-Spectrum-KSampler `1925664`](https://github.com/sorryhyun/ComfyUI-Spectrum-KSampler/blob/192566487d0d9cdbd892e464eedd53c4efa972c2/fsg.py)、[anima_lora `703b34b`](https://github.com/sorryhyun/anima_lora/blob/703b34b40e3b724b20bf1a006168023241c59584/docs/inference/fsg.md) 均为 MIT。拒绝理由是证据与适配：（1）论文只在 ε-prediction 的 SD2.1/SDXL/Hunyuan-DiT/DiT 上用 DDIM/DDPM 验证，且论文中的 FSG 是“前段长区间迭代 + 每步 CFG++ 校准”，其把迭代集中在最高噪声段的调度在 Anima 上会发散（σ≈0.94 时 ρ>1，anima_lora 作者探针）；（2）唯一的 Anima 证据是 anima_lora 作者自测：σ 频带 [0.59, 0.75] 只对应 28 步 er_sde、1024 档，随步数/分辨率移动（作者文档与代码对 20 步的频带说法还不一致）；已验证的组合是在 CFG++ λ=1.5 基底上叠 FSG，对照为单独 CFG++，只有目测 A/B，**同 NFE 对照从未做过**；作者已于 2026-07-12 关闭这条线，写明约 1.8× NFE 的 FSG 不如 1× NFE 的 `--xattn_boost`（已实现为 `sd_forge_anima_xattn_boost`，见上文），“不得宣传为免费画质”；（3）`remi` 没有经过 flow 数学验证的 CFG++（见上文 PPM 段），只移植 FSG 于普通 CFG 既不对应论文，也不对应作者的 Anima 验证点，一并移植 CFG++ 超出本轮范围；（4）Forge 侧可以借用 DCW 的 pre-denoiser 就地改写 latent 的接缝（限单次评估采样器、inpaint mask 下失效），额外前向从 pre-CFG hook 以 auxiliary 调用发出，但还需要把校准步强制为 Spectrum 实际前向，而多次评估采样器（Heun、PC3 探测）上“哪次评估算一步”没有定义。在出现 flow 模型上的同 NFE 对照（对照 FSG 与增加步数/NFE 相同的 CFG）之前不移植。
- [DyPE（Dynamic Position Extrapolation，ICML 2026）](https://arxiv.org/abs/2510.20766)：**deferred（2026-09 第十五轮评估；implement later，条件见下）**。训练无关、零额外前向：按当前去噪时间调整 RoPE 外推强度（高噪声段接近插值、低噪声段回到原生外推，`k_t = dype_scale · t^dype_exponent`），让 DiT 在远超训练分辨率时先对齐低频结构、后解出高频。`remi` 现状：`backend/nn/anima.py` 的 `VideoRopePosition3DEmb` 在 init 时固定 NTK 因子（`huggingface_guess/detection.py:198-200` 给 Anima 的 h/w 外推比 4.0、t 1.0，head_dim 128 时空间 θ 放大约 4.29 倍），嵌入按 `(T, H, W, device)` 缓存、与 sigma 无关，没有任何随时间或随分辨率的位置编码调整。来源与许可：[官方代码 `436fe61`](https://github.com/guyyariv/DyPE/tree/436fe612fb7695fa2d5dc32d63390c048ba6033c) 为 MIT，但**自 `3773c5d`（2025-10-26）起 README 写明 “This work is patent pending. For commercial use or licensing inquiries, please contact the authors”**，MIT 不含专利授权；官方只实现 FLUX 与 Qwen-Image（diffusers）。[wildminder/ComfyUI-DyPE `bbfd46a`（v2.16.0）](https://github.com/wildminder/ComfyUI-DyPE/tree/bbfd46ac99d898bcd35c1911fadced12e0635dd6) 为 Apache-2.0（其专利授权只覆盖该仓库贡献者，不覆盖论文作者的专利），[v2.4.0（2026-08-04）](https://github.com/wildminder/ComfyUI-DyPE/releases/tag/2.4.0) 起支持 `model_type=anima`（`src/models/anima.py` 按 Cosmos 的逐轴 θ 重建旋转矩阵，并修正了与内置 NTK 因子的重复外推）。推迟理由：（1）**收益区间在 `remi` 的常用尺寸之外**：官方模型卡给出的 Anima 可用范围是 512²–1536²（1536² ≈ 2.36 MP），DyPE 只在 token 网格超出基准网格时生效（`scale_global = max(1, span / base_patches)`），论文收益集中在训练分辨率数倍处（FLUX 训练约 1 MP，主评测与缩放分析在 2K–4K 以上，差距随分辨率增大）；Hires Guard 虽默认关闭（`anima_presets.py` 的 `hires_guard.enabled: False`），其 Balanced/Detail 预设把 Hires 目标限制在 2.1/2.36 MP，即全部落在 Anima 原生范围内，启用 Guard 时 DyPE 必为空操作；超过 1536² 的现有路径是 MultiDiffusion 分块，每块仍在原生范围内。（2）**没有任何 Anima 指标**：论文只在 FLUX、FiTv2（附录含 Qwen-Image）上评测；ComfyUI 移植没有 Anima 基准，README 称 Anima 的 `base_resolution` 为 “1920（auto-detected）”，但 `DyPE` 节点实际默认 1024 且不自动检测（只有 SEGA 读取 `max_img_h`），1024 基准会在 Anima 原生范围内（>1024 px 边长）就开始改写 RoPE；其 Anima 示例工作流用的是 `method=base`（关闭 DyPE），README 还提示 Anima 上非 `vision_yarn` 方法在 2K 以上会出现斑点噪声。（3）**Forge 侧代价**：嵌入需按每次调用的 sigma 重建（当前 RoPE 缓存与 per-block compile 都假定它只取决于网格），Spectrum 的特征预测会跨越随 sigma 变化的位置编码，PC3 辅助探测、Hires/img2img 起始 sigma 与 edit 参考帧（T 轴）都需要定义；这些都要在有收益证据后才值得做。**实施条件（全部满足才进入队列）**：① 在 Anima Base/Aesthetic 上做单次生成 2048² 及以上的固定 seed 对照（≥20 seeds，对照组为原生 RoPE、以及同耗时的 1536² + Hires/MultiDiffusion），结构错误统计或盲选显示稳定收益；② 专利状态澄清（作者对开源再分发/非商业使用给出书面许可，或专利未授权/放弃），在此之前不把该方法作为内置功能分发；③ 依论文（CC BY 4.0）clean-room 实现为默认关闭的 Anima 专用选项：基准网格取 1536²（96×96 patch）而非 1024，只在超出时生效，RoPE 缓存键加入量化后的 `k_t`，并与 Hires Guard 上限、Spectrum、SPEED、per-block compile 明确交互规则。本轮不写代码。
- [T-GATE（Cross-Attention Makes Inference Cumbersome）](https://arxiv.org/abs/2404.02747)：**rejected（2026-09 第十七轮评估）**。在“语义规划”阶段之后冻结 cross-attention 输出（缓存 cond/uncond 的均值）以省去后续步的 cross-attention。[官方代码 `ab2a88f`](https://github.com/HaozheLiu-ST/T-GATE/tree/ab2a88fa50319cbb0bfcdc2772405573b266fa13) 为 MIT，许可无障碍；论文评测 SD 系列 UNet 与 PixArt 等 DiT，没有 Anima。拒绝理由：（1）[anima_lora `703b34b` 的 `crossattn_self_attn_dominance.md`](https://github.com/sorryhyun/anima_lora/blob/703b34b40e3b724b20bf1a006168023241c59584/docs/findings/crossattn_self_attn_dominance.md) 测得 Anima 的 cross-attention 在每个 σ 都只占 attention 残差更新的一小部分（σ>0.9 约 15%，σ<0.45 约 6%，含 MLP 时占总更新约 5%→3.5%），可省的只是这一小项；同文 Result 1 显示 cross-attention map 从不锁定（低 σ 仍有约 0.2–0.3 的重路由率，由 self-attention 锐化后的 image query 带动），冻结不是恒等变换；（2）Spectrum 已在缓存步跳过整个前向，T-GATE 只能作用于实际前向步，且冻结值会进入 Spectrum 的特征历史；（3）`sd_forge_anima_xattn_boost` 在高噪声窗口改写 cross-attention 残差，FreeFuse/Regional 在 cross-attention 上路由/加掩码，冻结输出会让这些逐步行为在门控后失效，需要逐一定义交互。没有 Anima 上的同耗时对照前不移植。
- [anima_lora 的 deferred-foveated merge（token merging）](https://github.com/sorryhyun/anima_lora/blob/703b34b40e3b724b20bf1a006168023241c59584/docs/inference/foveated.md)：**rejected（2026-09 第十七轮评估）**。σ_c 以下把外围 2×2 token 平均为一个 token（1024²/28 步/CFG 4 下端到端 ×1.37）。作者已于 2026-07-03 归档这条线：外围软模糊是方法本身固有的（任何尾段处理都无法恢复），被判定不可接受；它还与 Spectrum 互斥（作者 bench P3 结论为“不要再提”），只支持 Euler。一般的 token merging（ToMe/ToMA 等）在 Anima 上同样会把合并区域的速度改为组共享，并改变 token 数（per-block compile 需要重编译，FreeFuse/Regional 的 token 级掩码需要重映射），在没有无模糊的 Anima 证据前不移植。
- [CADS（Condition-Annealed Diffusion Sampler，ICLR 2024）](https://arxiv.org/abs/2310.17347)：**deferred（2026-09 第十九轮评估）**。训练无关、零额外前向：高噪声段对条件向量加随时间衰减的高斯噪声 `ŷ = √γ(t)·y + s·√(1−γ(t))·n`（可选按原统计量重缩放 ψ），在高 CFG 下恢复同 prompt 多样性；论文在 SD 上加在 CLIP 文本嵌入上，ImageNet DiT 类条件 FID 1.70。来源：作者为 ETH Zurich 与 Disney Research|Studios，**没有官方代码**；社区实现 [v0xie/sd-webui-cads `11c7f1f`](https://github.com/v0xie/sd-webui-cads/tree/11c7f1f8dc3d579e202adcc6e736c6b3936defde) 为 GPL-3.0（与本仓库 AGPL-3.0 兼容）；本轮未检索到相关专利，但 Disney Research 的方法未见作者对专利状态的声明，不能视为已澄清。推迟理由：（1）与已实现的 DAVE 目标相同（同 prompt 的布局多样性），而 DAVE 有 Anima 上的作者 bench，CADS 没有任何 Anima/flow DiT 证据；（2）注入点不等价：Anima 的 context 是 Qwen3 → LLM adapter 输出、填充到 512 token 的序列，填充行是 attention sink（anima_lora `crossattn_self_attn_dominance.md` Result 1/4），对全部行加噪会改变 sink 行为，只对有效 token 加噪、以及 ψ 按 token 还是按序列重缩放都需要实验决定；（3）Forge 侧交互：每步不同的噪声 context 会进入 Spectrum 的逐分支特征历史（需要强制实际前向或把噪声纳入签名）、NAG 的正向 context、Regional/Artist Mixer 的独立 context（是否一并加噪）与 FreeFuse 的掩码收集步，均需逐一定义。**实施条件**：Anima Base 上固定 seed 对照（≥20 seeds，与 DAVE 同多样性指标、同提示词遵循度比较）显示 CADS 有 DAVE 之外的收益，且专利状态得到确认；届时依论文 clean-room 实现为默认关闭、只作用于条件分支有效 token 的选项。本轮不写代码。
- 加载时 4 bit / MXFP8 量化（NVFP4、MXFP8、ConvRot W4A4）：**deferred（2026-09 第二十轮评估）**。`remi` 现有的加载时量化只有 scaled FP8 与 INT8 ConvRot（`backend/nn/anima_quant.py`）。现状：上游 backend 已注册 `TensorCoreNVFP4Layout`、`TensorCoreMXFP8Layout` 与 `TensorCoreConvRotW4A4Layout`（`backend/quant_ops.py`），加载器按每层 `comfy_quant` 元数据识别预量化权重（`backend/state_dict.py`、`backend/operations_mixed_precision.py`），所以 [Comfy-Org/comfy-quants `3d057e9` 的 Anima 导出](https://github.com/Comfy-Org/comfy-quants/blob/3d057e9c8e3132ef1853dec96f91d3de0aaf0e6a/docs/quantization/anima.md)（FP8/MXFP8/NVFP4，量化 block 1..N−1 的 attention/MLP/AdaLN Linear，保留 block 0、block 1 的 AdaLN、final layer、embedder、LLM adapter 与全部 norm；Anima 的 in-features 与 16/32 分组对齐，无需填充）这类预量化 checkpoint 走的是现有加载路径，不需要新代码。推迟“加载时转换”的理由：（1）MXFP8/NVFP4 的原生 GEMM 只在 Blackwell（SM≥10）上可用，本仓库没有可做速度/显存/画质对照的硬件，CPU 只能验证 layout 往返，不能决定是否值得；（2）没有 Anima 上 4 bit round-to-nearest 的画质证据：comfy-quants 只给出导出流程，未给画质指标，且明确不为 Anima 提供 INT4（其 INT4 走 SVDQuant/AWQ 的混合路径，需要校准，不是无校准的加载时转换）；ConvRot 论文（arXiv:2512.03673）的 W4A4 结果不在 Anima 上；（3）FP8 路径已覆盖“全精度 checkpoint 省一半显存”的主要需求。**实施条件**：在 Blackwell 上对照 scaled FP8 与 comfy-quants 导出的 NVFP4/MXFP8 checkpoint（≥20 seeds 固定 seed 画质、峰值显存、每步耗时），加载时 RTN 转换与导出件画质一致且相对 FP8 有可测收益；届时沿用 comfy-quants 的 Anima 层选择与 `anima_quant` 的敏感层列表，作为“Diffusion in Low Bits”的新选项实现，并补 CPU layout 往返测试。本轮不写代码。
- [Entropy Rectifying Guidance（ERG，NeurIPS 2025，arXiv:2504.13987）](https://arxiv.org/abs/2504.13987)：**deferred（2026-10 第三十八轮评估）**。在推理时修改 DiT 的 attention（按熵调整）来构造引导所需的“弱”预测，论文称可同时改善画质、多样性与提示词一致性，并能与 CADS 等方法组合。推迟理由：本轮在常见位置与 GitHub 搜索中都没有找到官方代码，只能依论文 clean-room 实现；没有 Anima/Cosmos 上的证据；它要改的正是 NAG、FreeFuse、Regional 已在使用的 attention seam（`ANIMA_ATTENTION_MODIFIERS` 与 cross-attention 包装），需要定义叠加次序，且会进入 Spectrum 的特征历史。**实施条件**：出现可用许可的参考实现或足以复现的算法细节，并有 Anima Base 上与调好的 CFG/APG 同 NFE 的对照。
- [Where and How to Perturb（HeadHunter / SoftPAG，NeurIPS 2025，arXiv:2506.10978）](https://arxiv.org/abs/2506.10978)：**rejected（2026-10 第三十八轮评估）**。把 PAG 细化到单个 attention head：先用 PickScore/CLIPScore 等评分模型在多 GPU 上迭代搜索对某个目标有利的 head 组合（HeadHunter），再把选中 head 的 attention 图向单位矩阵线性插值（SoftPAG）。拒绝理由：[官方代码](https://github.com/cvlab-kaist/HeadHunter)（2026-10 核对）没有 LICENSE 文件，不能复制；选 head 需要针对每个模型用评分模型做大规模 GPU 搜索，论文只给出 SD3 与 FLUX.1-dev 的结果，没有 Anima 的 head 表；SoftPAG 本身仍是 PAG 式的额外前向，与上文 PAG 条目的推迟理由相同。

## 建议实施顺序与验证矩阵

1. **Anima NAG**（已实现）：独立 commit；先加入可组合 attention override，再实现 UI 与范围控制。
2. **Momentum Guidance 原型**（已实现）：从 Flow Euler 开始，证明状态不会被 PC3 auxiliary call、Spectrum 或 Hires 污染后再独立 commit。
3. **Anima per-block compile**（已实现）：独立 commit；作为现有 Compile 插件预设，明确这只是速度优化。
4. **FDG 实验模式**：先做无 `kornia` 的最小原型，并与 SMC/CFG-Zero*/APG 强制互斥（已实现；CFG-Zero* 与 FDG 同属一个模式下拉框）。
5. **Guidance active range**（已实现为 Anima Guidance 的 CFG active range）：仍待 Base/Aesthetic 的 GPU A/B。
6. **PiD GPU 验证**：已作为可选内置后处理器；补齐权重、显存、中断和批次实测。
7. **Layer Replay**：仅在 GPU 对照实验证明收益后 clean-room 实现。

每项质量功能至少覆盖：

- 模型：Base、Aesthetic、Turbo；
- 采样：官方 ER-SDE、Euler，现有 beta57、Flow Euler/UniPC2/PC3；
- 步数/CFG：Turbo 8/12 步 CFG 1；Base/Aesthetic 30/40 步 CFG 4.5；
- 尺寸：768²、1024²、1536²、非正方形；
- 功能组合：无扩展、Spectrum、LLLite、Regional、Artist Mixer、Hires、batch 2；
- 证据：固定 prompt/seed 的像素或 latent 差异、耗时/峰值显存、至少 20 seeds 的盲选或结构错误统计。

在完成这一矩阵前，“作者样图更好”“节点星数高”或单张图对比都不能证明对 Forge Neo 的普遍收益。
