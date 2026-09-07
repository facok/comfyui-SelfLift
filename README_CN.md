# comfyui-SelfLift（中文说明）

[English README](README.md)

在 ComfyUI 中提供渐进分辨率采样。图像节点实现论文 [SelfLift: Accelerating Few-Step Diffusion via Self-Recovering Resolution Transition](https://arxiv.org/abs/2609.02036) 中的 **SelfLift-zero**（Artifact-Aware Consistency Lift），用于兼容的 rectified-flow 图像骨干。MiniMax H3 节点是实验性的音视频适配；论文没有验证 MiniMax H3。

H3 节点提供两种不同模式：

- `upscaler_model=none` 且 `rho>0`：使用论文式 SelfLift-zero 过渡，即 nearest 直接提升加选择性的 pixel-VAE 锚点。
- 已安装 H3 upscaler 且 `rho=0`：使用学习式纯 latent 提升。这是实用默认模式，但既不是 SelfLift-zero，也不是 SelfLift-rich。

渐进分辨率推理让前期去噪步骤跑在低分辨率、收尾回到全分辨率，从而砍掉大部分模型评估的空间开销。在过渡点，先预测干净端点（式 3），然后做两种提升——直接 latent 上采样（`z_lat`）和解码 → 像素上采样 → 重新编码（`z_pix`）——两者的不一致构成一张局部伪影风险图，用来把高风险位置朝 VAE 可达的锚点修正（式 4–9），最后把修正后的估计在过渡 sigma 处重新加噪（式 10），调度在全分辨率下继续。

## 可选 H3 upscaler

从 [LBH-123-AI/Minimax_h3_latent_Upscaler](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler) 下载 checkpoint，放入 `ComfyUI/models/latent_upscale_models/`，然后重启 ComfyUI。H3 节点默认选择检测到的第一个文件名包含 `h3` 的模型；如果没有找到，则默认使用 `none`。

## 节点

两个节点使用与 `SamplerCustom` 相同的 `sampler`/`sigmas` 接口。`KSamplerSelect` 必须选择标准 `euler`，调度器仍使用模型原有配置。其他 sampler 会被拒绝，因为拆分多步、祖先或 SDE 求解器会重置历史状态或改变随机过程。

- **SelfLift Progressive Sampler (MiniMax H3)**（`sampling/minimax`）：H3 音视频 latent（如 *Empty MiniMax H3 AV Latent*）。音频流没有空间维度，会使用复用的 Euler 边界预测继续推进而不做空间提升；关键帧条件 latent 会在前缀阶段同步缩放到低分辨率网格。这是工程扩展，不是论文验证过的配置。
- **SelfLift Progressive Sampler (Image)**（`sampling`）：4D 图像 latent（如 *Empty Latent Image*）。

### 参数

- `transition_step`：在低分辨率执行的 denoiser evaluation 数量。它用于选择论文的过渡边界 `t_r`，但自身是步骤数，不是连续时间值。对于 `N` 步调度，有效范围是 `1` 到 `N-1`。论文的 4-step FLUX.2-Klein 使用 `3`，8-step Z-Image-Turbo 使用 `6`。
- `lowres_scale`：前缀的空间缩放（论文：0.5 = ¼ token）。
- `rho`：向 pixel-VAE 锚点修正的最高风险位置比例。`rho=0` 会跳过 pixel route 和全部 artifact-aware correction。
- `w_min`、`w_max`：选中 mask 内的修正范围，必须满足 `0 <= w_min <= w_max <= 1`。论文使用 `0.5 / 1.0`；当 `rho=1` 且 `w_min=w_max=1` 时是纯像素锚点。
- `latent_upsample`（仅图像节点）：直接提升的插值方式。论文使用 `nearest`；`bilinear` 是可选实验。
- `upscaler_model`（仅 H3 节点）：学习式 3D 卷积直接提升器。`none` 使用 nearest-neighbor。外部模型与 `rho>0` 同时使用属于混合实验，不是论文定义的 SelfLift-zero。
- `seed`、`cfg`、`sampler` 和 `sigmas` 与 `SamplerCustom` 语义相同。只接受 `s_churn=0` 的标准 Euler。

| 节点 | `transition_step` | `lowres_scale` | `rho` | `w_min / w_max` | 直接提升 |
| --- | ---: | ---: | ---: | ---: | --- |
| MiniMax H3 | 6 | 0.5 | 0 | `0.5 / 1.0` | 第一个已安装 H3 upscaler，否则 nearest |
| Image | 6 | 0.5 | 0.3 | `0.5 / 1.0` | nearest |

图像节点默认值对应论文的 8-step Z-Image-Turbo 配置。使用 4-step FLUX.2-Klein 时，应把 `transition_step` 改为 `3`，并把 `rho` 改为 `0.4`。

转换不会增加 denoiser evaluation。最后一次低分辨率 Euler 评估直接提供式 3 的预测；提升并重新加噪后，同一预测继续完成该 Euler 区间。包含 `N` 步的 schedule 因而仍严格保持 `N` NFE：其中 `transition_step` 次在低分辨率，其余在目标分辨率。除非 `rho=0` 跳过 pixel route，SelfLift-zero 还会增加一次 VAE 解码 → 缩放 → 编码往返。安装 H3 checkpoint 后，H3 默认走外部纯 latent 路径。要运行 SelfLift-zero，请选择 `upscaler_model=none` 并设置 `rho>0`。

## 阶段计时与过渡内存

两个节点都会输出 `[SelfLift timing]` 日志，记录低分辨率采样、过渡阶段（端点准备、成对提升、修正及调试输出、重新加噪）和高分辨率采样。采样日志包含逐步耗时和阶段总耗时。首步包含采样器及模型准备；回调之间的计时包含预览和前一步的 Euler 更新。阶段总耗时还包含最终更新、清理，以及高分辨率阶段的输出传输。这些是墙钟时间，不是独立的 GPU kernel 耗时。

默认计时不强制同步 CUDA。需要同步诊断时，在启动 ComfyUI 前设置 `SELFLIFT_TIMING_SYNC=1`，即可在计时边界同步模型所在的 CUDA 设备；这可能减少执行重叠，正常使用时无需设置。CPU 执行不会调用 CUDA 同步。

过渡时提前完成音频边界更新，在提升前释放无用的低分辨率状态；像素锚点的解码帧会在 VAE 编码前释放。模型驻留仍由 ComfyUI 管理，这些改动只释放普通张量，不强制卸载模型或清空 CUDA 缓存。

## 适用范围

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
