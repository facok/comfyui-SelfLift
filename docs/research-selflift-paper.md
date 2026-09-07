# SelfLift 论文实现核对基准

本文记录对 arXiv:2609.02036v1 一手论文及其 TeX 源码的核查结果，重点覆盖可用于审查本仓库 Progressive Sampler 的 SelfLift-zero 公式、Algorithm 1、采样语义和适用边界。

## 一手来源

- Tingyan Wen et al., **SelfLift: Accelerating Few-Step Diffusion via Self-Recovering Resolution Transition**, arXiv:2609.02036v1, 2026-09-02: <https://arxiv.org/abs/2609.02036v1>
- 论文 PDF: <https://arxiv.org/pdf/2609.02036v1>
- arXiv TeX 源码: <https://arxiv.org/src/2609.02036v1>
- 作者项目页（由 arXiv 元数据直接链接）: <https://happygirlty.github.io/SelfLift_res/>

除明确标为“实现推论”的内容外，下文公式和结论均来自论文正文第 3 节、附录 A/B/C/E 和 Algorithm 1。

## 结论先行

1. 论文中的 SelfLift-zero 是为 **rectified-flow 图像模型**定义的单次分辨率转换：先在低分辨率推进到 `t_r`，从当前状态预测低分辨率 clean endpoint，再分别进行 latent 提升和 decode-resize-encode，按二者差异在空间上选择 top-`rho` 位置并局部校正，之后用一份新采样的高分辨率高斯噪声在原 flow marginal 上重新加噪，最后沿原时间表完成高分辨率采样。[正文 Sec. 3.1-3.3; Eq. (1)-(10); Appendix A, Algorithm 1]
2. 论文明确声称 SelfLift-zero **不增加 denoiser evaluation，也不修改采样 schedule**。因此实现若在低分辨率前缀结束后额外调用一次模型来得到 Eq. (3)，其数学结果可以与公式一致，但 NFE 和性能口径不再等同于论文；严格复现应复用前缀最后一次模型评估的速度/去噪预测，或把前缀边界组织为同一次评估。[摘要; 正文 Sec. 3.3; Appendix A, Algorithm 1]
3. 论文没有 MiniMax H3 实验、音频 latent 公式、3D 时空 latent 权重规则，也没有 H3 latent upscaler。主实验只覆盖 FLUX.2-Klein-9b 和 Z-Image-Turbo 图像模型；视频只在附录 E 给出 Wan2.1-T2V 的初步负面观察。因此“MiniMax H3 SelfLift”属于工程外推，不能声称是论文验证过的实现。[正文 Sec. 4.1; Appendix E]
4. 论文的 direct latent route 明确使用 **nearest-neighbor latent lifting**；pixel route 是低分辨率解码、像素空间放大、目标分辨率重新编码。[正文 Eq. (4)-(5); Appendix B]
5. artifact risk 是每个样本内部、每个空间位置跨 `C` 通道的平均 L1 差异。top-`rho` 选择和 min-max 归一化也必须逐样本计算；不能把 batch、通道、帧或音频维混入同一个分位数统计，除非明确承认这是视频扩展策略。[正文 Eq. (7)-(8)]
6. Eq. (10) 要求重新采样 `xi ~ N(0,I)` 并构造 `(1-sigma_t_r) * z0_hat_H + sigma_t_r * xi`。直接放大已有 noisy latent，或把原低分辨率噪声插值成高分辨率噪声，都不是论文 Algorithm 1。[正文 Eq. (10); Appendix A, Algorithm 1]

## 数学定义

论文基于 rectified flow。干净 latent `z_0` 与高斯噪声 `epsilon` 的边际为：

```text
z_t = (1 - t) z_0 + t epsilon,       epsilon ~ N(0, I)
v_theta(z_t, t, c) ~= E[epsilon - z_0 | z_t, c]
```

采样从 `t=1` 向 `t=0` 进行。令 `Phi^R_(a->b)` 表示分辨率 `R` 下从时间 `a` 到 `b` 的数值流，渐进分辨率采样写为：

```text
z_0^PR = Phi^H_(t_r->0)( T_(L->H)( Phi^L_(1->t_r)(epsilon^L; c) ); c )
```

其中 `T_(L->H)` 是 SelfLift 要实现的跨分辨率转换算子。[正文 Eq. (1)-(2)]

### 1. 预测 clean endpoint

在低分辨率状态 `z^L_(t_r)` 上：

```text
z0_hat^L_(t_r) = z^L_(t_r) - t_r * v_theta(z^L_(t_r), t_r, c)
```

这是 rectified-flow 参数化下的 clean endpoint 预测。代码若使用 ComfyUI 的 `sigma` 参数化，必须确认传给模型和用于恢复 clean sample 的量与模型 sampling wrapper 的定义一致，不能仅凭变量名把 `sigma` 当作论文的 `t`。[正文 Eq. (3)]

### 2. 构造两条配对提升路径

两条路径必须从同一个 `z0_hat^L_(t_r)` 出发：

```text
z_lat = U_lat(z0_hat^L_(t_r))

Y     = U_pix(D_L(z0_hat^L_(t_r)))
z_pix = E_H(Y)
```

`z_lat` 保留低分辨率轨迹的确定性对应关系，但可能落在目标 VAE latent 分布不支持的子空间；`z_pix` 是目标编码器可达的稳定 anchor，但会损失不可从放大图像唯一恢复的高频细节。因此论文保留 `z_lat` 为主体，只用 `z_pix` 做局部校正，而不是全局替换或固定比例混合。[正文 Eq. (4)-(6), Observation 1/2]

附录 B 在 1,000 个 FLUX.2-Klein 校准样本上报告目标 VAE decode-encode round-trip energy：native HR latent 为 `0.01894`，pixel-VAE route 为 `0.02717`（native 的 `1.43x`），direct latent lift 为 `0.35859`（native 的 `18.93x`）。这支持“pixel route 稳定但偏平滑、latent route 锐利但可能不一致”的设计。[Appendix B, Eq. (17) and Table 6]

### 3. Artifact-Aware Consistency Lift

先计算差值：

```text
Delta = z_pix - z_lat
```

对样本中的每个空间位置 `i`，跨 `C` 个 latent 通道计算：

```text
s_i = (1 / C) * ||Delta_i||_1
M_i = 1[ s_i >= Q_(1-rho)({s_j}_j) ]
```

`Q_(1-rho)` 是同一样本所有空间位置分数的 `(1-rho)` 分位数，所以 `rho` 表示要校正的最高风险空间位置比例。选区内计算 `s_min`、`s_max`，得到：

```text
W_i = M_i * [
    w_min + (w_max - w_min)
    * (s_i - s_min) / (s_max - s_min + epsilon)
]
```

最终 clean latent 为：

```text
z0_hat^H_(t_r)
    = z_lat + W * Delta
    = (1-W) * z_lat + W * z_pix
```

由公式可得几个实现边界：

- `W` 是空间权重，广播到 latent 通道；原论文对象是 `C x H x W` 图像 latent。
- `rho=0` 在数学上意味着空选区，需要实现显式短路，避免分位数或空集合 `min/max` 异常。
- `rho=1` 选择全部位置；当 `w_min=w_max=1` 时结果严格等于 `z_pix`。
- 分数相同导致阈值 ties 时，`>= quantile` 可能选择超过 `rho` 的位置，这是原公式本身允许的行为。
- 若选区只有相同分数，分母靠 `epsilon` 保持有限，此时所有选中位置权重接近 `w_min`。

[正文 Eq. (7)-(9)]

### 4. 在转换时刻重新加噪

论文不是直接继续使用提升后的 clean latent，而是重新建立目标分辨率下、同一转换时刻的 flow marginal：

```text
xi ~ N(0, I)
z^H_(t_r) = (1 - sigma_tilde_(t_r)) * z0_hat^H_(t_r)
            + sigma_tilde_(t_r) * xi
```

`xi` 的形状必须与目标高分辨率 latent 一致。这里论文把 schedule 写成成对的 `(t_k, sigma_tilde_tk)`；Eq. (3) 使用 `t_r`，Eq. (10) 使用 schedule 的 `sigma_tilde_(t_r)`。实现不能无证明地假设两者在任意模型 wrapper 下相同。[正文 Eq. (10); Appendix A, Algorithm 1]

## Algorithm 1 的精确采样流程

Algorithm 1 的 SelfLift-zero 分支可逐项转写为：

1. 选用原始预训练模型 `v_theta0`。
2. 采样低分辨率初始噪声 `z^L_1 ~ N(0,I)`。
3. 按原 schedule 在低分辨率从 `1` rollout 到 `t_r`，得到 `z^L_(t_r)`。
4. 用 Eq. (3) 得到 `z0_hat^L_(t_r)`。
5. 用 Eq. (4)-(5) 构造 `z_lat` 和 `z_pix`。
6. 计算 `Delta`、risk map `s`、top-`rho` mask `M` 和权重 `W`。
7. 用 Eq. (9) 得到校正后的 `z0_hat^H_(t_r)`。
8. 独立采样目标形状噪声 `xi`，用 Eq. (10) 得到 `z^H_(t_r)`。
9. 从同一个 schedule 的 `t_r` 继续高分辨率 rollout 到 `0`。
10. 用高分辨率 decoder `D_H` 解码最终 `z^H_0`。

论文把 schedule 作为输入且明确“unchanged few-step schedule”。转换不是额外插入一个从 `t_r` 到下一时刻的特殊 solver step；它替换该边界上的状态，然后正常完成余下轨迹。[Appendix A, Algorithm 1]

## 论文实验参数

主实验均为 `1024 x 1024` 输出、`512 x 512` 低分辨率前缀、单次 `2x` 空间转换：

| Backbone | NFE | `t_r` | `rho` | `[w_min, w_max]` |
| --- | ---: | ---: | ---: | ---: |
| FLUX.2-Klein-9b | 4 | 3 | 0.4 | `[0.5, 1.0]` |
| Z-Image-Turbo | 8 | 6 | 0.3 | `[0.5, 1.0]` |

附录消融说明：较小 `rho` 更保锐但可能残留 artifact；较大 `rho` 更稳定但更平滑。提高 `w_min` 能加强校正，适合更激进的转换或文字渲染，但可能降低清晰度。论文默认 `[0.5,1.0]` 是稳定性与锐度之间的平衡。[正文 Sec. 4.1; Appendix D.1]

## 对 MiniMax H3 实现的适用边界

论文没有把以下行为定义为 SelfLift-zero 的组成部分：

- 音频 latent 在转换处如何处理或如何与视频噪声耦合；
- 对 `C x T x H x W` 视频 latent，risk score 是逐帧空间计算、跨时间聚合，还是把 `T*H*W` 共同做 quantile；
- 对 keyframe、首尾帧、mask、参考图条件 latent 如何同步缩放；
- 用学习式 H3 latent upscaler 替代论文的 nearest-neighbor direct route；
- 对 H3 的 VAE normalization、patch/grid 对齐、原生分辨率集合和 sigma parameterization 的具体规则。

因此这些都应在代码和 README 中标明为 **针对 H3 的适配策略或经验性变体**，不能归因于论文。尤其是 learned latent upscaler：论文的 SelfLift-rich lifter 是 10.8M 参数的 SwinIR-style `2x` 图像 latent lifter，输入 128 通道，由 SelfLift-zero target 蒸馏训练；它不是 MiniMax H3 的通用 3D latent upscaler。[正文 Eq. (11)-(12), Sec. 4.1; Appendix C]

附录 E 只说明该原理“可以”扩展到视频，但要求 backbone 可靠支持所选分辨率。作者在 Wan2.1-T2V 上发现把 480p 输入空间下采样 `2x` 会改变 token sequence 分布并破坏场景结构，因而建议视频渐进分辨率推理只在模型原生多分辨率范围内使用。对 MiniMax H3，至少应单独验证低分辨率网格是模型原生支持的、时长/token 序列语义不变、VAE 在两种空间分辨率上兼容。[Appendix E]

## 代码审查清单

审查节点实现时应逐条确认：

- 前缀确实运行在缩小后的空间 latent 上，转换只发生一次，转换后从同一 schedule 边界继续。
- Eq. (3) 使用的是 solver 边界上真实的 noisy state，而不是某个 sampler 为继续采样而缩放/封装后的表示。
- Eq. (3) 的模型输出被正确解释为 rectified-flow velocity 或等价 clean prediction，时间量与模型 wrapper 一致。
- clean probe 是否复用已有模型评估；若额外评估，README 的 NFE、延迟和“论文复现”表述应明确区别。
- latent route 与 pixel route 从完全相同的 clean estimate 出发。
- paper mode 的 latent route 是 nearest-neighbor，pixel route 使用与 backbone latent 空间匹配的同一 VAE family。
- `s_i` 是通道平均 L1，quantile、选区 `min/max` 按 batch sample 独立计算。
- `rho=0`、`rho=1`、`w_min=w_max` 和常量 risk map 均无 NaN、空张量或错误选区。
- `W` 只在定义的空间/时空位置变化，并正确广播到通道。
- Eq. (10) 使用目标 latent 形状的新标准高斯噪声和转换时刻的正确 schedule sigma。
- 高分辨率后缀没有跳过或重复 `t_r` 对应的 solver interval。
- H3 的 audio latent、条件 latent、mask/keyframe latent 和视频 latent 在切换前后保持形状与语义一致。
- H3 使用的所有非论文设计均有单独测试和说明，尤其是 3D learned upscaler、时间维 risk 聚合及原生分辨率限制。

## 仓库修复状态

1. README 和 BibTeX 已使用 arXiv v1 的正式标题 “SelfLift: Accelerating Few-Step Diffusion via Self-Recovering Resolution Transition”。
2. 独立 Eq. 3 probe 已移除。实现现在复用最后一次低分辨率 Euler evaluation 的 clean prediction，在提升和重新加噪后用同一预测完成该 Euler 区间，因此总 NFE 与输入 schedule 一致。
3. 节点只接受 rectified-flow 模型和标准 Euler sampler，避免分段调用破坏 multistep history 或随机过程。
4. H3 默认值已恢复为局部 Artifact-Aware Consistency Lift；旧 probe 路径得到的纯 pixel-anchor 经验结论已删除。README 明确将 H3、音频续接、时空 risk 和 learned upscaler 标为论文外工程扩展。
