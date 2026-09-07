# comfyui-SelfLift

[中文说明](README_CN.md)

Progressive-resolution sampling in ComfyUI, implementing **SelfLift-zero**
(Artifact-Aware Consistency Lift) from
[SelfLift: Accelerating Few-Step Diffusion via Self-Recovering Resolution Transition](https://arxiv.org/abs/2609.02036)
for compatible rectified-flow image backbones, with an experimental **MiniMax H3**
audio-video adaptation. The paper evaluates FLUX.2-Klein and Z-Image-Turbo; it does
not evaluate MiniMax H3.

Progressive-resolution inference runs the early denoising steps at low resolution and
finishes at full resolution, cutting the spatial cost of most model evaluations.
At the transition, the predicted clean endpoint (Eq. 3) is lifted two ways — direct
latent upsampling (`z_lat`) and decode → pixel upscale → re-encode (`z_pix`) — and
their disagreement becomes a localized artifact-risk map that corrects high-risk
locations toward the VAE-reachable anchor (Eqs. 4–9), then the corrected estimate is
re-noised at the transition sigma (Eq. 10) and the schedule resumes at full resolution.

## Nodes

Both nodes use `sampler`/`sigmas` inputs like `SamplerCustom`. Connect the standard
`euler` sampler from `KSamplerSelect` and the model's normal scheduler. Other samplers
are rejected because splitting multistep, ancestral, or SDE solvers would reset solver
history or alter their stochastic process.

- **SelfLift Progressive Sampler (MiniMax H3)** (`sampling/minimax`): H3 AV latents
  (e.g. from *Empty MiniMax H3 AV Latent*). The audio stream has no spatial
  dimensions and continues the reused Euler boundary step without spatial lifting;
  keyframe condition latents are rescaled to the low-res grid for the prefix. This is
  an engineering extension, not a configuration validated by the paper.
- **SelfLift Progressive Sampler (Image)** (`sampling`): 4D image latents
  (e.g. *Empty Latent Image*).

Parameters:

- `transition_step` (`t_r`): number of denoiser evaluations executed at low
  resolution. The paper uses 3 of 4 and 6 of 8 NFEs on FLUX.2-Klein /
  Z-Image-Turbo.
- `lowres_scale`: spatial scale of the prefix (paper: 0.5 = ¼ tokens).
- `rho`, `w_min`, `w_max`: correction blend (paper: 0.3–0.4 / 0.5 / 1.0).
  `rho=0` = plain direct lift; `rho=w=1.0` = pure pixel anchor. H3 defaults to
  the paper's 8-NFE correction parameters (`0.3 / 0.5 / 1.0`), not a global anchor.
- `latent_upsample`: interpolation of the direct lift (paper: nearest).
- `upscaler_model` (H3 node only): a learned 3D-conv latent upscaler
  (e.g. [LBH-123-AI/Minimax_h3_latent_Upscaler](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler),
  placed under `models/latent_upscale_models/`). This optional external model is an
  H3-specific experiment and is not the paper's SelfLift-rich lifter. The default
  `none` uses nearest-neighbor lifting as specified by SelfLift-zero.

The transition does not add a denoiser evaluation. The final low-resolution Euler
evaluation supplies Eq. 3; after lifting and re-noising, its prediction also completes
that Euler interval. A schedule with `N` steps therefore remains exactly `N` NFEs:
`transition_step` at low resolution and the rest at target resolution. SelfLift-zero
adds one VAE decode → resize → encode round trip unless `rho=0` skips the pixel route.

## Applicability

Use the VAE belonging to the sampled model so the pixel anchor remains in the same
latent space. The paper requires the backbone to support both selected resolutions.
Its preliminary Wan2.1 video experiment found that unsupported token sequence lengths
destabilized structure, so H3 resolutions, temporal behavior, and quality must be
validated independently. Previous H3/Krea2 measurements from the probe-based
implementation are intentionally not retained because they do not describe this
corrected NFE-equivalent path.

## Not included

SelfLift-rich (the distilled latent lifter + On-Policy Self Recovery) requires
training and is outside this plugin.

## Reference

```
@article{wen2026selflift,
  title={SelfLift: Accelerating Few-Step Diffusion via Self-Recovering Resolution Transition},
  author={Wen, Tingyan et al.},
  journal={arXiv:2609.02036},
  year={2026}
}
```
