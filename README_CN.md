# comfyui-SelfLift（中文说明）

[English README](README.md)

在 ComfyUI 中实现渐进分辨率采样，即论文
[SelfLift: Accelerating Few-Step Diffusion via Self-Recovering Resolution Transition](https://arxiv.org/abs/2609.02036)
中的 **SelfLift-zero**（Artifact-Aware Consistency Lift），用于兼容的 rectified-flow
图像骨干，并提供实验性的 **MiniMax H3** 音视频适配。论文只验证了 FLUX.2-Klein 和
Z-Image-Turbo，没有验证 MiniMax H3。

渐进分辨率推理让前期去噪步骤跑在低分辨率、收尾回到全分辨率，从而砍掉大部分模型评估的空间开销。
在过渡点，先预测干净端点（式 3），然后做两种提升——直接 latent 上采样（`z_lat`）和
解码 → 像素上采样 → 重新编码（`z_pix`）——两者的不一致构成一张局部伪影风险图，
用来把高风险位置朝 VAE 可达的锚点修正（式 4–9），最后把修正后的估计在过渡 sigma 处
重新加噪（式 10），调度在全分辨率下继续。

## 节点

两个节点使用与 `SamplerCustom` 相同的 `sampler`/`sigmas` 接口。`KSamplerSelect`
必须选择标准 `euler`，调度器仍使用模型原有配置。其他 sampler 会被拒绝，因为拆分
多步、祖先或 SDE 求解器会重置历史状态或改变随机过程。

- **SelfLift Progressive Sampler (MiniMax H3)**（`sampling/minimax`）：H3 音视频
  latent（如 *Empty MiniMax H3 AV Latent*）。音频流没有空间维度，会使用复用的 Euler
  边界预测继续推进而不做空间提升；关键帧条件 latent 会在前缀阶段同步缩放到低分辨率
  网格。这是工程扩展，不是论文验证过的配置。
- **SelfLift Progressive Sampler (Image)**（`sampling`）：4D 图像 latent
  （如 *Empty Latent Image*）。

参数：

- `transition_step`（`t_r`）：在低分辨率执行的 denoiser evaluation 数量。论文在
  FLUX.2-Klein / Z-Image-Turbo 上分别使用 4 NFE 中的 3 次、8 NFE 中的 6 次。
- `lowres_scale`：前缀的空间缩放（论文：0.5 = ¼ token）。
- `rho`、`w_min`、`w_max`：修正混合权重（论文：0.3–0.4 / 0.5 / 1.0）。
  `rho=0` = 纯直接提升；`rho=w=1.0` = 纯像素锚点。H3 默认使用论文 8-NFE 的
  局部修正参数（`0.3 / 0.5 / 1.0`），不再使用全局像素锚点。
- `latent_upsample`：直接提升的插值方式（论文：nearest）。
- `upscaler_model`（仅 H3 节点）：学习式 3D 卷积 latent 提升器
  （如 [LBH-123-AI/Minimax_h3_latent_Upscaler](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler)，
  放到 `models/latent_upscale_models/` 下）。这是可选的 H3 外部实验模型，不是论文中的
  SelfLift-rich lifter。默认 `none` 使用 SelfLift-zero 规定的 nearest-neighbor 提升。

转换不会增加 denoiser evaluation。最后一次低分辨率 Euler 评估直接提供式 3 的预测；
提升并重新加噪后，同一预测继续完成该 Euler 区间。包含 `N` 步的 schedule 因而仍严格
保持 `N` NFE：其中 `transition_step` 次在低分辨率，其余在目标分辨率。除非 `rho=0`
跳过 pixel route，SelfLift-zero 还会增加一次 VAE 解码 → 缩放 → 编码往返。

## 适用范围

必须使用采样模型所属的 VAE，保证像素锚点仍位于同一 latent 空间。论文要求 backbone
可靠支持所选的两种分辨率；其 Wan2.1 初步视频实验发现，不受支持的 token sequence
长度会破坏场景结构。因此 H3 的分辨率、时序行为和质量需要独立验证。旧 probe 实现下
得到的 H3/Krea2 测量结果没有保留，因为它们不能代表当前修正后的 NFE 等价路径。

## 未包含

SelfLift-rich（蒸馏 latent 提升器 + On-Policy Self Recovery）需要训练，
不在本插件范围内。

## 引用

```
@article{wen2026selflift,
  title={SelfLift: Accelerating Few-Step Diffusion via Self-Recovering Resolution Transition},
  author={Wen, Tingyan et al.},
  journal={arXiv:2609.02036},
  year={2026}
}
```
