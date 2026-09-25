# Anima 生成期优化技术与 Forge Neo 移植调研

> 调研日期：2026-08-09
> 对照分支：`remi`，基线提交 `d940ce41`
> 范围：只评估生成期质量、提示词/空间控制、采样、加速和高分辨率后处理；训练技术不作为本报告的主要对象。

## 结论

### Forge Neo 落地状态（2026-08-09）

本报告中进入移植队列的项目已按默认关闭、生成级隔离和明确互斥的原则集成到 `remi` 工作区：

- NAG 通过 Anima 原生 attention modifier chain 接入，不覆盖 Regional、Artist Mixer 或 FreeFuse 的 wrapper；合成的 Regional/Artist context 与 NAG 隔离，低显存导致正负分支无法合批时会明确记录 inactive；
- Momentum Guidance 在 flow velocity 空间维护每次生成独立的 EMA，限于单次评估的 Euler/Anima Flow Euler/FreeFuse Euler，FreeFuse probe 后会重置状态；
- Anima per-block compile 作为现有 Compile 内置插件的互斥预设，调用期临时替换 block，并在 `finally` 中恢复；Windows 先预检 C++ 编译器，Dynamo/Inductor 懒编译失败则回退 eager；
- FDG 作为实验 CFG controller，使用纯 PyTorch 两层频带分解，不增加 Kornia 硬依赖，并与 DCW/CNS 显式互斥；
- Guidance Active Range 使用归一化采样进度映射当前 scheduler sigma；
- 现有 PiD 后处理器修复了类级结果泄漏和多批次 prompt/seed 错位，并在 GUI 中明示 Hires 互斥与权重许可限制。

目前已完成 CPU 单元/组合回归、Ruff、`compileall` 和 `git diff --check`；仍需在含 Anima 权重的 NVIDIA 环境执行报告末尾的 GPU 固定 seed 矩阵，才能对画质收益和编译加速比例作实证结论。

当前 `remi` 并不是“缺少 Anima 优化”的状态。官方明确推荐的 beta57、原生 flow Euler 路径、ER-SDE、控制条件、区域提示、LoRA 调度、缓存/预测、FreeFuse 和高分辨率防护等主要能力已经内置。继续盲目预装 ComfyUI 节点，更多会造成重复 patch、不可组合的 attention hook 和难以解释的 UI，而不是提高质量。

下一轮建议严格按以下顺序推进：

1. **P0：移植 Anima NAG（Normalized Attention Guidance）**。它最直接补足 Turbo/CFG 1 下负面提示词控制，论文和 Anima 专用节点都有公开实现，MIT 许可清晰。Forge 需要增加一个可组合的 Anima cross-attention override 接口，不能直接照搬 ComfyUI 的 `optimized_attention_override`。
2. **P1 实验：Momentum Guidance**。它针对 flow model，以跨 timestep 的速度 EMA 做外推，不增加模型前向；和 Anima 的 flow 轨迹匹配，但必须先隔离 PC3、Spectrum、SMC 的状态。
3. **P1：把 Anima 逐 block `torch.compile` 做成现有内置 Compile 插件的预设**。这是速度优化，不改变生成算法；来源为 MIT，改动小，且比另装一个插件更符合 Forge 的无感集成方式。
4. **P2 实验：Frequency-Decoupled Guidance（FDG）**。它允许低/高频使用不同 guidance 强度，适合研究低 CFG 细节，但没有 Anima 实证、占用 CFG combine，并会引入 `kornia`，不应默认开启。

其后才是 **PiD 4× 解码器**（效果潜力高但很重）和 **Layer Replay**（效果证据弱、原仓库缺许可证，只能 clean-room 重写）。ANIMA_BOOSTER 与 FLSampler 均为保留全部权利的代码，不能复制或移植；Anima IP-Adapter 当前仓库则同时存在许可证、权重和代码可运行性问题，暂不应集成。

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
| 编译加速 | `sd_forge_compile` 整体模型 `torch.compile` | 已有，但缺 Anima 逐 block 预设 |
| NegPiP | 内置 Neo 适配扩展 | 已有；无需复制 ComfyUI-ppm 的版本 |
| PiD 4× 后处理 | `backend/diffusion_engine/pid.py` + `sd_forge_pid` | 已可从当前 latent 一键进入 4-step PiD；权重需用户预先安装，与 Hires 互斥 |

## 候选技术与优先级

### P0：Anima NAG —— 最值得移植的质量功能

[NAG 论文](https://arxiv.org/abs/2505.21179)把正负条件的外推从最终噪声预测移到 cross-attention 输出，并用 L1 范数限制外推幅度，再与原正向 attention 混合。论文报告其在少步数场景中改善负面提示控制、文本对齐和感知质量；这是作者报告的跨模型结果，**不是 Anima 专属 GPU 基准**。

[ComfyUI-Anima-NAG 固定版本](https://github.com/hybskgks28275/ComfyUI-Anima-NAG/tree/15904013437882197b8ef7dd69891e79105bf0fb)是 [MIT 许可](https://github.com/hybskgks28275/ComfyUI-Anima-NAG/blob/15904013437882197b8ef7dd69891e79105bf0fb/LICENSE)的 Anima 专用适配。其核心过程是：

1. 计算正常的正/负批次 cross-attention 输出；
2. 取正向 query 与负向 key/value 再计算一次 `z_neg`；
3. `z_tilde = z_pos + scale * (z_pos - z_neg)`；
4. 用 `tau` 对 `||z_tilde||₁ / ||z_pos||₁` 做上限约束；
5. 用 `alpha` 混合回原正向 attention。

对 Turbo 的建议起点是 `scale=2.0, tau=2.5, alpha=0.5`，仅在前 50% 步数启用。它会增加启用区间内的 cross-attention 计算，但比完整 CFG 双分支外推更有针对性。

**Forge 移植接口：** 当前 `backend/nn/anima.py::SelfCrossAttention` 直接调用全局 `attention_function`，没有 ComfyUI 的 `optimized_attention_override`。应先在 Anima attention 层增加可链式 override，并明确只作用于 cross-attention。区域提示和 Artist Mixer 都会临时包装 `cross_attn.forward`；NAG 必须位于其下层的 Q/K/V attention 接口，才能让区域上下文分别获得正确的负向引导，且不能通过永久 monkeypatch 覆盖它们。

**必须测试的组合：** NAG × Regional、Artist Mixer、LLLite、SageAttention、Spectrum、batch > 1、Hires 二阶段、CFG 1/4.5、Turbo/Base。若没有同时包含正负分支，NAG 应明确无效果并给出提示。

### P1：Anima 逐 block `torch.compile` —— 低风险加速（排序在 Momentum Guidance 之后）

[ComfyUI-Anima-BlockCompile 固定版本](https://github.com/sorryhyun/ComfyUI-Anima-BlockCompile/tree/455eeeaeda63f1985eaa1669e4bd08cf3ec4bd13)使用 [MIT 许可](https://github.com/sorryhyun/ComfyUI-Anima-BlockCompile/blob/455eeeaeda63f1985eaa1669e4bd08cf3ec4bd13/LICENSE)，逐个编译 `diffusion_model.blocks.{i}`，而不是把整个 DiT 包成一个巨大 graph。这样可缩短首次编译、减少 graph break，并复用结构相同的 transformer block 图。

Forge 已有 `extensions-builtin/sd_forge_compile/scripts/compile.py`，因此不应新增独立扩展。建议增加 `Anima per-block` 预设：

- 仅当模型确认为 Anima 且存在非空 `diffusion_model.blocks` 时启用；
- 对每个 block 延迟 `torch.compile`，使用现有 guard filter 排除 `transformer_options`；
- 与整模型 compile 互斥，并能完整恢复原 module；
- Windows 预检 Inductor 所需 C++ 编译器，失败时回退 eager，而不是在首张图中途报错；
- 分别验证固定/动态分辨率、LoRA 切换、SageAttention 和 Hires 两阶段。

这是速度功能，不应宣称提高画质；生成结果需要做相同 seed 的数值/视觉回归，防止编译后精度路径变化。

### P1 实验：Momentum Guidance —— flow 方向匹配，但先解决状态隔离

[Momentum Guidance 论文](https://arxiv.org/abs/2602.20360)面向 flow model，用历史速度的指数移动平均对当前 guidance 速度做外推，不增加额外模型前向；作者在 SD3、FLUX.1-dev 等 flow 模型上报告了收益。已有 [MIT 许可的 ComfyUI 实现](https://github.com/pamparamm/sd-perturbed-attention/blob/904319bff623b185f15047e231a446d615ee48c6/mg_nodes.py)，核心只依赖 PyTorch。从模型类型看它比传统 diffusion-only 技巧更值得在 Anima 上试验，但目前没有 Anima 专属基准。

它与 SMC 不同：SMC 修正 cond/uncond 控制误差，Momentum Guidance 在合并后的跨 timestep 轨迹维护 EMA；二者同时启用的数学含义未定义。Forge 原型必须在新任务、batch/shape 变化、Hires 二阶段时重置，且 PC3/UniPC 辅助前向和 Spectrum 预测/缓存都不能推进主轨迹 EMA。建议只从 Flow Euler 固定 seed A/B 开始，不默认启用。

### P2 实验：Frequency-Decoupled Guidance（FDG）

[FDG 论文](https://arxiv.org/abs/2506.19713)把 cond-uncond guidance update 经 Laplacian pyramid 拆成低频与高频分量：低频保留结构/提示词约束，高频使用独立强度增强细节。论文报告它在低 CFG 下提高 fidelity 并减轻高 CFG 的过饱和；[MIT 节点实现](https://github.com/pamparamm/sd-perturbed-attention/blob/904319bff623b185f15047e231a446d615ee48c6/fdg_nodes.py)额外依赖 `kornia`。

它没有 Anima 实测证据，并且会完全接管 CFG combine，不能直接与 SMC、CFG-Zero* 或 APG 同时启用。它和 DCW/CNS 都涉及频域，但作用点不同：FDG 分解同一步的 guidance residual，DCW/CNS 修正跨步误差或噪声；无定义叠加仍可能过锐。上游还会把 latent padding 到下一次方尺寸，非标准长宽比会增加显存。建议将其作为 Guidance controller 的互斥实验模式，若进入 Forge 核心则自行实现小型 Gaussian/Laplacian 分解，避免把 `kornia` 变成硬依赖。

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

它对普通 CFG 模型有明确研究依据，但 Turbo 本来就是 CFG 1，因此对 Turbo 没意义；对 Anima Base/Aesthetic 可能降低过饱和并改善分布质量。PPM 默认 `sigma_start=5.42/sigma_end=0.28` 来自非 flow schedule，不能用于 Anima 的 `0..1` flow sigma；Forge 应存储归一化采样进度，再映射到实际 schedule。当前 Guidance 扩展已有多种 CFG 控制，最佳集成方式是增加一个“CFG active range”，而不是预装整个 PPM 节点包。需要先以 Base/Aesthetic、CFG 4–7 做 A/B。

PPM 的 AttentionCouple 和 NegPiP 已分别被当前 Regional 与 NegPiP 覆盖；AttentionCouple 会再次占用 attention/mask patch seam，与 Regional 同时启用还会产生未定义的 hook 顺序、mask resize 和 Spectrum cache 语义，因此不移植。动态采样器和 CFG++ 也不能未经 flow/const 数学验证直接套到 Anima。

### P3：Dynamic CFG / 频域细节 —— 只适合作为研究原型

[ComfyUI-AnimaDynamicCFG 固定版本源码](https://github.com/DanrisiUA/ComfyUI-AnimaDynamicCFG/blob/35b70e0c402ea52e60aa18aafd87cce6ba3b0c55/nodes.py)包含 CFG 曲线、自适应 CFG、CFG rescale、latent 均值校正、FFT 高频增强和后期随机噪声，但仓库当前**没有 LICENSE 文件**，因此不能复制、修改或分发其源码。它也没有测试和量化证据，当前实现还存在几项方法学问题：

- 自适应 CFG 使用 `cond-uncond` 的**绝对整体 L2 norm**，随分辨率和 batch 改变，默认 `target_divergence=1` 没有跨尺寸意义；
- 进度依赖 `sample_sigmas`，缺失时退化成固定 1→0，未证明符合 Anima 的实际 sigma；
- 高频放大和随机噪声可能只是制造锐化/颗粒感，也可能破坏 flow 轨迹；随机噪声使用全局 `torch.randn_like`，没有绑定 sampler generator，固定 seed 也不保证可复现；
- “更写实”与官方将 Anima 定义为非写实模型的目标并不一致。

其中标准 CFG Rescale 已有论文依据，但其余功能不应直接移植；scheduled CFG、rescale 和频域修正也分别与现有 Stage Scheduler、Prompt Rescale、DCW/CNS 部分重叠。若研究，只能依据公开算法 clean-room 重写，先把 divergence 改成每元素 RMS/robust norm，并加入确定性 seed 与 flow-aware 进度。

### P3：Anima IP-Adapter —— 方向重要，当前仓库不可交付

[comfyui-anima-ipadapter](https://github.com/Wenaka2004/comfyui-anima-ipadapter) 宣称以 SigLIP2/InstantCharacter 风格的 decoupled cross-attention实现参考图人物一致性。这是 Anima 生态真正缺少的重要方向，但当前仓库不能移植：

- 没有 LICENSE，不能复制代码；
- README 没有可下载的训练完成权重，[issue #2](https://github.com/Wenaka2004/comfyui-anima-ipadapter/issues/2)也明确询问 Anima checkpoint 位置而没有得到可用发布；
- 当前源码中 `TransformerEncoderLayer` 被以不支持的 `memory=` 参数调用，loader 还向 `IPAdapterSigLIP` 传入不存在的 `input_dim=` 参数；
- cross-attention wrapper 的函数签名与当前 Forge/Comfy Anima 路径不一致；
- patch 生命周期、逐步 timestep 更新和卸载逻辑不完整。

因此应等待上游发布许可、权重和可运行版本，或取得明确授权后重新设计。不能把训练脚本或空壳节点预装给用户。

## 已有或不应移植的项目

### 已经覆盖

- [ComfyUI-Anima-LLLite](https://github.com/kohya-ss/ComfyUI-Anima-LLLite)：Apache-2.0，但当前 Forge 已内置更完整的 Anima LLLite v2，包括图像/掩码、时间范围和多控制组合。
- [TeaCache](https://github.com/welltop-cn/ComfyUI-TeaCache) / [TeaCache 论文](https://arxiv.org/abs/2411.19108)：属于 timestep-aware 缓存；当前 Spectrum 已做 Anima 专用预测与 SEA。可以借鉴评测指标，但再叠缓存会造成双重跳步和状态冲突。
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

- [CFG-Zero*](https://arxiv.org/abs/2503.18886)：针对 flow matching 的早期 zero-init 和优化 guidance scale；在 Flux/SD3 等模型上有结果，适合与现有 Flow controls 做独立数学复核。
- [On the Guidance of Flow Matching](https://arxiv.org/abs/2502.02150)：给出一般 flow guidance 框架，更适合作为校验现有/未来引导算法的理论参考。
- [Token Perturbation Guidance（TPG）论文](https://arxiv.org/abs/2506.10036)及其 [MIT 官方实现](https://github.com/TaatiTeam/Token-Perturbation-Guidance)：对中间 token 做保持范数的 shuffle，以原始/扰动预测之差产生 guidance。官方只验证 SDXL、SD2.1；Anima 需要选定 block、增加额外预测，并会与 Spectrum、FreeFuse、Modulation 共用 block/attention seam，只适合开发者研究开关。
- [Perturbed Attention Guidance（PAG）论文](https://arxiv.org/abs/2403.17377)及 [Diffusers 官方管线](https://huggingface.co/docs/diffusers/api/pipelines/pag)：理论上可作用于 Anima self-attention，但没有 Anima 实证，会增加预测成本，并与 FreeFuse/Spectrum 的 attention/cache 路径高度敏感，优先级低于 NAG、MG、FDG。
- [Rectified-CFG++](https://rectified-cfgpp.github.io/)是 flow 模型的 predictor-corrector guidance；其求解路径与已经移植的 KeithZ117 Flow Corrective / PC3 高度重叠。除非同一基准证明它有独立收益，否则不再加入第二套 predictor-corrector sampler。
- [Energy-based Rectified Flow Guidance](https://arxiv.org/abs/2504.13987)、[Fine-Grained Perturbation Guidance](https://arxiv.org/abs/2506.10978)：仍需先确认官方代码许可、额外前向次数和 Anima attention/flow 参数化是否匹配。

## 建议实施顺序与验证矩阵

1. **Anima NAG**：独立 commit；先加入可组合 attention override，再实现 UI 与范围控制。
2. **Momentum Guidance 原型**：从 Flow Euler 开始，证明状态不会被 PC3 auxiliary call、Spectrum 或 Hires 污染后再独立 commit。
3. **Anima per-block compile**：独立 commit；作为现有 Compile 插件预设，明确这只是速度优化。
4. **FDG 实验模式**：先做无 `kornia` 的最小原型，并与 SMC/CFG-Zero*/APG 强制互斥。
5. **Guidance active range 原型**：先本地实验，不默认启用；证据充分后再 commit。
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
