# comfyui-SelfLift（中文说明）

[English README](README.md)

在 ComfyUI 中提供渐进分辨率采样。图像节点实现论文 [SelfLift: Accelerating Few-Step Diffusion via Self-Recovering Resolution Transition](https://arxiv.org/abs/2609.02036) 中的 **SelfLift-zero**（Artifact-Aware Consistency Lift），用于兼容的 rectified-flow 图像骨干。MiniMax H3 节点是实验性的音视频适配；论文没有验证 MiniMax H3。

H3 节点提供两种不同模式：

- `upscaler_model=none` 且 `rho>0`：使用论文式 SelfLift-zero 过渡，即 nearest 直接提升加选择性的 pixel-VAE 锚点。
- 已安装 H3 upscaler 且 `rho=0`：使用学习式纯 latent 提升。这是实用默认模式，但既不是 SelfLift-zero，也不是 SelfLift-rich。

渐进分辨率推理让前期去噪步骤跑在低分辨率、收尾回到全分辨率，从而砍掉大部分模型评估的空间开销。在过渡点，先预测干净端点（式 3），然后做两种提升——直接 latent 上采样（`z_lat`）和解码 → 像素上采样 → 重新编码（`z_pix`）——两者的不一致构成一张局部伪影风险图，用来把高风险位置朝 VAE 可达的锚点修正（式 4–9），最后把修正后的估计在过渡 sigma 处重新加噪（式 10），调度在全分辨率下继续。

## 可选 H3 upscaler

加载器会将浮点权重、偏置和归一化张量统一到输入卷积的精度。输入权重为 FP8 时，若 checkpoint 中存在 BF16 张量则采用 BF16，否则采用 FP16。纯 FP16/BF16/FP32 模型保留原精度。转换发生在向 ComfyUI 模型管理器注册之前。

从 [LBH-123-AI/Minimax_h3_latent_Upscaler](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler) 下载 checkpoint，放入 `ComfyUI/models/latent_upscale_models/`，然后重启 ComfyUI。H3 节点默认选择检测到的第一个文件名包含 `h3` 的模型；如果没有找到，则默认使用 `none`。

## 高分辨率分块（实验性）

H3 节点的 `highres_tiling`（高分辨率分块）开关默认关闭。开启后，在高分辨率采样准备时，将空闲显存与同设备上 ComfyUI 管理的驻留权重相加（模型去重、总量不超过设备容量），按 ComfyUI 的最低权重驻留比例留出权重空间（不超过当前模型大小），再扣除 ComfyUI 的最低推理预留，作为工作区估算目标。这样不会把可回收的模型权重误判为永久占用；实际加载、驻留和卸载仍由 ComfyUI 决定。依次评估 1–8 块并选择首个满足目标的方案。空间方向按 2×2 patch 对齐后的长轴选择，最小核心区域为两个 patch；极小 latent 使用整图。达到分块上限仍不满足目标时会记录 estimate_fits=false，不保证避免 OOM。低分辨率阶段和过渡流程不变。

关键帧随目标区域裁切，R2V 参考保持完整，保留全图位置坐标。每块都输入完整音频，但只保留首块的音频预测。这不会跳过其他块的音频计算，也未证明比融合音频更好。分块间没有全局注意力，可能降低激活显存，但可能改变构图、细节及音画一致性，也不保证提速。该模式不支持 ControlNet。

开启分块不会增加 Euler 更新次数或进度回调数，但每次高分辨率评估需要与块数相同的空间块模型前向（尚未计入 CFG）；本文的原始 NFE 数量说明适用于关闭分块时。

采样准备按最大的补齐空间块估算工作内存，同时计入完整音频、参考条件、文本及裁切后的关键帧，并为采样和融合预留八份完整 FP32 latent 缓冲的余量。它沿用 ComfyUI 的 H3 内存估算公式，不是实测峰值或硬上限；实际 latent、条件和 sigma 调度不会因此缩小。加载、驻留、卸载以及额外模型的预留仍由 ComfyUI 管理。`[SelfLift tiling memory]` 会记录最大块和额外预留前的最低/首选预算；极小 latent 保持原准备路径。

## 节点

两个节点使用与 `SamplerCustom` 相同的 `sampler`/`sigmas` 接口。`KSamplerSelect` 必须选择标准 `euler`，调度器仍使用模型原有配置。其他 sampler 会被拒绝，因为拆分多步、祖先或 SDE 求解器会重置历史状态或改变随机过程。

- **SelfLift Progressive Sampler (MiniMax H3)**（`selflift`）：H3 音视频 latent（如 *Empty MiniMax H3 AV Latent*）。音频流没有空间维度，会使用复用的 Euler 边界预测继续推进而不做空间提升；关键帧条件 latent 会在前缀阶段同步缩放到低分辨率网格。这是工程扩展，不是论文验证过的配置。
- **SelfLift Progressive Sampler (Image)**（`selflift`）：4D 图像 latent（如 *Empty Latent Image*）。
- **H3 Temporal State Transport (TST)**（`selflift`）：`MODEL` → `MODEL` 补丁节点，把免训练的 Temporal State Transport 校正（TST，[arXiv:2609.08505](https://arxiv.org/abs/2609.08505)）应用到 H3 的联合 packed 注意力上。可配合任意标准采样节点（KSampler/SamplerCustom），不限于 SelfLift 采样器。详见下方专门章节。

### 参数

- `transition_step`：在低分辨率执行的 denoiser evaluation 数量。它用于选择论文的过渡边界 `t_r`，但自身是步骤数，不是连续时间值。对于 `N` 步调度，有效范围是 `1` 到 `N-1`。论文的 4-step FLUX.2-Klein 使用 `3`，8-step Z-Image-Turbo 使用 `6`。
- `lowres_scale`：前缀的空间缩放（论文：0.5 = ¼ token）。
- `rho`：向 pixel-VAE 锚点修正的最高风险位置比例。`rho=0` 会跳过 pixel route 和全部 artifact-aware correction。
- `w_min`、`w_max`：选中 mask 内的修正范围，必须满足 `0 <= w_min <= w_max <= 1`。论文使用 `0.5 / 1.0`；当 `rho=1` 且 `w_min=w_max=1` 时是纯像素锚点。两者均为零时，无论 `rho` 为何，都会跳过像素路径和修正。
- `latent_upsample`（仅图像节点）：直接提升的插值方式。论文使用 `nearest`；`bilinear` 是可选实验。
- `upscaler_model`（仅 H3 节点）：学习式 3D 卷积直接提升器。`none` 使用 nearest-neighbor。外部模型与 `rho>0` 同时使用属于混合实验，不是论文定义的 SelfLift-zero。
- `seed`、`cfg`、`sampler` 和 `sigmas` 与 `SamplerCustom` 语义相同。只接受 `s_churn=0` 的标准 Euler。

| 节点 | `transition_step` | `lowres_scale` | `rho` | `w_min / w_max` | 直接提升 |
| --- | ---: | ---: | ---: | ---: | --- |
| MiniMax H3 | 6 | 0.5 | 0 | `0.5 / 1.0` | 第一个已安装 H3 upscaler，否则 nearest |
| Image | 6 | 0.5 | 0.3 | `0.5 / 1.0` | nearest |

图像节点默认值对应论文的 8-step Z-Image-Turbo 配置。使用 4-step FLUX.2-Klein 时，应把 `transition_step` 改为 `3`，并把 `rho` 改为 `0.4`。

转换不会增加 denoiser evaluation。最后一次低分辨率 Euler 评估直接提供式 3 的预测；提升并重新加噪后，同一预测继续完成该 Euler 区间。包含 `N` 步的 schedule 因而仍严格保持 `N` NFE：其中 `transition_step` 次在低分辨率，其余在目标分辨率。除非 `rho=0` 或两个修正权重均为零，SelfLift-zero 还会增加一次 VAE 解码 → 缩放 → 编码往返。安装 H3 checkpoint 后，H3 默认走外部纯 latent 路径。要运行 SelfLift-zero，请选择 `upscaler_model=none`，设置 `rho>0` 并使用非零修正权重。

## H3 Temporal State Transport (TST)

[TST](https://github.com/lytang63/temporal-state-transport) 到 MiniMax H3 的实验性移植。原论文只在 Wan2.2 上验证，未验证 H3；本节点是工程适配，不是经过验证的配置。TST 不盲目加强时间注意力，而是先做诊断：碎片化传输（注意力质量集中在过少帧上）会让细节漂移，过度混合（质量过于均匀地摊开）会破坏运动物理。

H3 没有独立的时间注意力——它是在 `[text | cond/refs | audio | video]` 打包序列上的单流 transformer——因此节点对 video 段（永远是最后一个打包段）的 post-RoPE query/key 做帧内空间均值池化，逐头构造帧级传输算子 `A`（F×F），计算谱张力 `T = H_row - H_vN`（论文式 1–3），并只对 video 行的 query 施加稳态温度 `γ = exp(τ_eff · T)`（论文式 5–6）。张力为正则锐化，为负则软化。`τ_eff` 沿用论文余弦调度：更深的层和更早的去噪步校正更强。text、audio、reference 行从不被缩放。干预通过 ComfyUI 的 `optimized_attention_override` 钩子实现，与注意力后端无关（sage/flash/SDPA/kitchen 均可）。整个流程零训练、零权重改动、无可学习参数；额外开销只是每次注意力调用上的几个 F×F 小矩阵运算（实测每次模型前向 0.1–0.3 秒，约占采样时间 1–2%）。

- `tau`：校正强度，论文默认 `0.2`；`0` 关闭校正但保留诊断。
- `log_diagnostics`：每次模型前向输出一行 `[H3 TST]` 日志，包含步号、latent 帧数、每帧网格行数、平均张力绝对值 `|T|`、平均 `γ`、以及 `|γ-1| > 0.03` 的头比例。逐头张力在每次调用自身校正之前测量；跨步的变化反映之前调用和之前步的校正效果。

帧数和每帧网格行数按前向捕获，SelfLift 采样器的低/高分辨率两个阶段各自正确处理。层号来自实际视频注意力调用计数；步号用当前 sigma 在 `sample_sigmas` 中匹配，不依赖任何会在 CFG 或阶段拆分下失效的调用计数假设。

一次单工作流、单 seed 的扫描（Ref2VA、5 秒、9-step Euler、CFG 1）发现：该内容上张力净值为**正**（过度混合方向，与 Wan 上报告的碎片化主导相反）；基线 `|T|` 从首步 0.449 上升到末步 0.486；剂量响应曲线在 `tau=0.2` 处测得 `|T|` 最低。`tau=0.5` 时 `|T|` 反超基线，画面中的文字和细节可见退化，与论文对过大 `tau` 的警告一致。这只是单一 prompt 单一 seed 的结果；请把 `0.2` 当作起点，按内容重新验证。

限制：TST **与 `highres_tiling` 不兼容**——wrapper 按注册顺序执行，本节点只能看到分块 wrapper 之外的完整分辨率形状，tile 注意力调用会被长度守卫拦截，TST 跳过并在控制台给出警告。池化原型算子是精确 video→video 注意力质量的近似；缩放 video query 也会同时改变它对 text/audio 列的注意力（论文的 Wan 目标有独立时间注意力，不存在这种泄漏）。已在真实 H3 运行上完成校准（启动 ComfyUI 前设置 `SELFLIFT_TST_EXACT=1`，会在每次前向的中间层调用上计算逐位置精确质量算子，输出 `[H3 TST exact]` 日志）：原型与精确张力的逐头符号一致率 89–100%，原型 `|T|` 与精确值的偏差约 10% 以内、略带高估——校正方向可靠，强度是近似的。

## 阶段计时与过渡内存

`[SelfLift plan]` 记录舍入后的低分辨率和目标 latent 形状、空间提升倍率、低/高分辨率 NFE、过渡预测和恢复采样的 sigma、CFG 及启用的提升路径。这些是 latent 尺寸，不是解码后的像素尺寸。`[SelfLift upscaler]` 记录模型文件、精度、尺度嵌入、分块/重叠长度、窗口数量、实际最大输入窗口和保守工作区预算。窗口包含填充和重叠，单位为 latent 时间位置，不是视频帧。`estimated_workspace` 是交给 ComfyUI 的启发式预算，不是实测峰值或显存硬上限；模型权重由管理器另外计算。

两个节点都会输出 `[SelfLift timing]` 日志，记录低分辨率采样、过渡阶段（端点准备、成对提升、修正及调试输出、重新加噪）和高分辨率采样。采样日志包含逐步耗时和阶段总耗时。首步包含采样器及模型准备；回调之间的计时包含预览和前一步的 Euler 更新。阶段总耗时还包含最终更新、清理，以及高分辨率阶段的输出传输。这些是墙钟时间，不是独立的 GPU kernel 耗时。

进度与过渡状态捕获使用各阶段的本地回调计数，因此外部封装带偏移的步号不会改变过渡位置。每次 Euler 评估仍须对应一次回调；缺失或多余的回调会给出明确错误。

默认计时不强制同步 CUDA。需要同步诊断时，在启动 ComfyUI 前设置 `SELFLIFT_TIMING_SYNC=1`，即可在计时边界同步模型所在的 CUDA 设备；这可能减少执行重叠，正常使用时无需设置。CPU 执行不会调用 CUDA 同步。

在启动 ComfyUI 前设置 `SELFLIFT_MEMORY_LOG=1`，可启用 `[SelfLift memory]`，在采样/过渡边界及 upscaler 加载和推理前后记录内存快照。日志以 MiB 为单位，包含进程 RSS、系统可用内存；CUDA 已初始化时还包含 PyTorch allocated/reserved、分配器峰值及设备已用/可用显存。设备占用包含其他分配和进程；`process_peak_allocated` 是分配器自上次重置以来的峰值，**不是当前阶段峰值**，边界快照可能漏掉中间瞬时峰值。日志不会重置共享统计、强制同步 CUDA 或改变模型驻留。CPU 执行只记录主存。开启观测有少量开销，对比运行时应保持日志设置一致。

保存中间 PNG 需要在启动 ComfyUI 前设置 `SELFLIFT_DEBUG=1`。节点会自动创建 `debug/`；仅保留该目录不再触发调试解码。调试输出可能解码完整中间视频，但只保存首帧，即使 `rho=0` 也会显著增加时间和内存开销。正常使用时不要设置此选项。调试文件已加入 Git 忽略规则。

过渡时提前完成音频边界更新，在提升前释放无用的低分辨率状态；像素锚点的解码帧会在 VAE 编码前释放。模型驻留仍由 ComfyUI 管理，这些改动只释放普通张量，不强制卸载模型或清空 CUDA 缓存。

## 适用范围

`latent_image` 必须是全零的尺寸模板，包括音频流。编码后的初始 latent 和 `noise_mask` 会在采样前被拒绝；此输入不支持图生图或局部重绘。通过 conditioning 传入的 H3 关键帧和参考条件仍然受支持。

`sigmas` 必须是一维浮点张量，各值有限、非负且单调不增。只有最后一个 sigma 可以为零，高分辨率起始 sigma（`sigmas[transition_step]`）必须小于 1。空调度或仅一个值的调度直接返回原输入；实际采样至少需要两步，`lowres_scale` 必须在 0.25 到 1 之间。非零末尾 sigma 保留采样器的部分去噪行为。`lowres_scale=1` 仍然执行过渡和重新加噪；全分辨率基线请使用原生采样器。

必须使用采样模型所属的 VAE，保证像素锚点仍位于同一 latent 空间。论文要求 backbone 可靠支持所选的两种分辨率；其 Wan2.1 初步视频实验发现，不受支持的 token sequence 长度会破坏场景结构。因此 H3 的分辨率、时序行为和质量需要独立验证。H3 标准的 768 像素短边在 `lowres_scale=0.5` 时会变成 384 像素，即使过渡实现正确，也可能超出 backbone 的训练分布。旧 probe 实现的结果不适用于当前修复后的 NFE 等价路径。

### 诊断 H3

固定同一 prompt 和 seed，使用 `transition_step=6`、`lowres_scale=0.5`：

| 测试 | `upscaler_model` | `rho` | 权重 | 含义 |
| --- | --- | ---: | --- | --- |
| 直接路径 | `none` | 0 | 任意 | 只使用 nearest-neighbor latent 提升 |
| 像素路径 | `none` | 1 | `1 / 1` | 只使用 H3 VAE 像素锚点 |
| 论文式 SelfLift-zero | `none` | 0.3 | `0.5 / 1` | 论文图像参数，风险图是实验性视频扩展 |
| H3 强修正 SelfLift-zero | `none` | 0.6 | `1 / 1` | H3 实测起点；过度平滑时再降低 |
| 外部提升器 | H3 checkpoint | 0 | 任意 | 学习式 H3 提升，不是 SelfLift-zero |

如果纯像素路径干净、局部修正不干净，问题在 H3 的时空风险图、mask 或修正参数；如果纯像素路径也有伪影，说明 H3 的过渡端点/VAE 往返不能提供 SelfLift-zero 所需的稳定互补锚点。如果只有外部提升器干净，实用方案就是渐进采样加该学习式提升器，不应称为 SelfLift-zero。

修复后的 H3 单 seed 对照实验（同一参考提示词、5 秒、8-step simple Euler）中，原生全分辨率基线和纯 pixel anchor 都是干净的；nearest 路径产生了大范围描边/油画化伪影，论文的图像参数（`rho=0.3`、`w_min=0.5`）仍保留大部分伪影。将 `rho` 提高到 0.6 后明显改善；`rho=0.6` 且 `w_min=w_max=1`，以及 `rho=1` 且 `w_min=0.5,w_max=1`，都消除了主要伪影。单 seed 结果不能证明 H3 的普遍兼容性，也不能作为通用参数；但它说明 SelfLift-zero 至少能在该配置上运行。本次实验中，H3 的直接提升误差比论文图像模型更广泛，所以论文的稀疏修正参数不能直接迁移。测试 H3 SelfLift-zero 时，建议从 `upscaler_model=none`、`rho=0.6`、`w_min=w_max=1` 开始；若过度平滑，再逐步降低修正强度。外部 learned lifter 仍是实用默认方案，但它不是 SelfLift-zero。

## 未包含

SelfLift-rich（蒸馏 latent 提升器 + On-Policy Self Recovery）需要训练，不在本插件范围内。

## 引用与致谢

- SelfLift 论文：[SelfLift: Accelerating Few-Step Diffusion via Self-Recovering Resolution Transition](https://arxiv.org/abs/2609.02036)
- TST 论文与代码：[Temporal State Transport in Video Generation](https://arxiv.org/abs/2609.08505)，[lytang63/temporal-state-transport](https://github.com/lytang63/temporal-state-transport)
- 可选 MiniMax H3 latent upscaler 权重与下载：[LBH-123-AI/Minimax_h3_latent_Upscaler](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler)
- 原始 ComfyUI 集成与推理实现：[LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler)

感谢 LBH-123-AI 公开 MiniMax H3 latent upscaler 权重和 ComfyUI 实现，使本插件能够提供可选的 H3 学习式提升路径。该外部 lifter 与 SelfLift 论文中的 SelfLift-rich 模型仍是彼此独立的实现。

```
@article{wen2026selflift,
  title={SelfLift: Accelerating Few-Step Diffusion via Self-Recovering Resolution Transition},
  author={Wen, Tingyan et al.},
  journal={arXiv:2609.02036},
  year={2026}
}
```
