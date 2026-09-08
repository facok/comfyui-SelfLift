# comfyui-SelfLift

[中文说明](README_CN.md)

Progressive-resolution sampling in ComfyUI. The image node implements **SelfLift-zero** (Artifact-Aware Consistency Lift) from [SelfLift: Accelerating Few-Step Diffusion via Self-Recovering Resolution Transition](https://arxiv.org/abs/2609.02036) for compatible rectified-flow image backbones. The MiniMax H3 node is an experimental audio-video adaptation; the paper does not evaluate MiniMax H3.

The H3 node exposes two distinct modes:

- `upscaler_model=none` with `rho>0` uses the paper-style SelfLift-zero transition: nearest-neighbor direct lift plus a selective pixel-VAE anchor.
- An installed H3 upscaler with `rho=0` uses a learned, latent-only lift. This is the practical default, but it is neither SelfLift-zero nor SelfLift-rich.

Progressive-resolution inference runs the early denoising steps at low resolution and finishes at full resolution, cutting the spatial cost of most model evaluations. At the transition, the predicted clean endpoint (Eq. 3) is lifted two ways — direct latent upsampling (`z_lat`) and decode → pixel upscale → re-encode (`z_pix`) — and their disagreement becomes a localized artifact-risk map that corrects high-risk locations toward the VAE-reachable anchor (Eqs. 4–9), then the corrected estimate is re-noised at the transition sigma (Eq. 10) and the schedule resumes at full resolution.

## Optional H3 upscaler

The loader aligns floating-point weights, biases and normalization tensors to the input convolution's dtype. For FP8 input weights, it uses BF16 when BF16 tensors are present in the checkpoint, otherwise FP16. Uniform FP16/BF16/FP32 checkpoints retain their precision. Conversion happens before registration with ComfyUI's model manager.

Download the checkpoint from [LBH-123-AI/Minimax_h3_latent_Upscaler](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler) and place it under `ComfyUI/models/latent_upscale_models/`, then restart ComfyUI. The H3 node selects the first detected filename containing `h3`; if none is found, the default is `none`.

## Nodes

Both nodes use `sampler`/`sigmas` inputs like `SamplerCustom`. Connect the standard `euler` sampler from `KSamplerSelect` and the model's normal scheduler. Other samplers are rejected because splitting multistep, ancestral, or SDE solvers would reset solver history or alter their stochastic process.

- **SelfLift Progressive Sampler (MiniMax H3)** (`sampling/minimax`): H3 AV latents (e.g. from *Empty MiniMax H3 AV Latent*). The audio stream has no spatial dimensions and continues the reused Euler boundary step without spatial lifting; keyframe condition latents are rescaled to the low-res grid for the prefix. This is an engineering extension, not a configuration validated by the paper.
- **SelfLift Progressive Sampler (Image)** (`sampling`): 4D image latents (e.g. *Empty Latent Image*).

### Parameters

- `transition_step`: number of denoiser evaluations executed at low resolution. This selects the paper's transition boundary `t_r`; it is a step count, not the literal continuous-time value. Valid values are `1` through `N-1` for an `N`-step schedule. Use `3` for the paper's 4-step FLUX.2-Klein setting and `6` for its 8-step Z-Image-Turbo setting.
- `lowres_scale`: spatial scale of the prefix (paper: 0.5 = ¼ tokens).
- `rho`: fraction of the highest-risk locations corrected toward the pixel-VAE anchor. `rho=0` skips the pixel route and all artifact-aware correction.
- `w_min`, `w_max`: correction range inside the selected mask. Values must satisfy `0 <= w_min <= w_max <= 1`. The paper uses `0.5 / 1.0`; `rho=1` with `w_min=w_max=1` is the pure pixel anchor. Setting both weights to zero skips the pixel route and correction, regardless of `rho`.
- `latent_upsample` (image node only): direct-lift interpolation. The paper uses `nearest`; `bilinear` is an optional experiment.
- `upscaler_model` (H3 node only): learned 3D-conv direct lifter. `none` uses nearest-neighbor lifting. Using an external model with `rho>0` is a hybrid experiment, not paper-defined SelfLift-zero.
- `seed`, `cfg`, `sampler`, and `sigmas` follow `SamplerCustom` semantics. Only standard Euler with `s_churn=0` is accepted.

| Node | `transition_step` | `lowres_scale` | `rho` | `w_min / w_max` | Direct lift |
| --- | ---: | ---: | ---: | ---: | --- |
| MiniMax H3 | 6 | 0.5 | 0 | `0.5 / 1.0` | First installed H3 upscaler, otherwise nearest |
| Image | 6 | 0.5 | 0.3 | `0.5 / 1.0` | Nearest |

The image defaults match the paper's 8-step Z-Image-Turbo setup. For the 4-step FLUX.2-Klein setup, change `transition_step` to `3` and `rho` to `0.4`.

The transition does not add a denoiser evaluation. The final low-resolution Euler evaluation supplies Eq. 3; after lifting and re-noising, its prediction also completes that Euler interval. A schedule with `N` steps therefore remains exactly `N` NFEs: `transition_step` at low resolution and the rest at target resolution. SelfLift-zero adds one VAE decode → resize → encode round trip unless `rho=0` or both correction weights are zero. With an H3 checkpoint installed, the H3 defaults take the latent-only external path. Select `upscaler_model=none` and set `rho>0` with nonzero correction weights to run SelfLift-zero.

## Timing and transition memory

`[SelfLift plan]` records the actual low/target latent shapes after rounding, spatial lift ratios, low/high NFE counts, transition prediction/resume sigmas, CFG and enabled lift routes. These are latent dimensions, not decoded pixel dimensions. `[SelfLift upscaler]` records the checkpoint, dtype, scale embedding, chunk/overlap, number of windows, largest actual input window and conservative workspace budget. Window lengths include padding and overlap and are measured in latent time positions, not video frames. `estimated_workspace` is a heuristic passed to ComfyUI, not measured peak memory or a hard cap; model weights are accounted for separately by the manager.

Both nodes log `[SelfLift timing]` messages for low-resolution sampling, the transition (endpoint preparation, paired lifts, correction/debug output, and re-noising), and high-resolution sampling. Sampling logs include each step and the stage total. The first step includes sampler/model preparation; callback intervals include previews and the preceding Euler update. The stage total also includes final updates, cleanup, and, for the high-resolution stage, output transfer. These are wall-clock measurements, not isolated GPU kernel timings.

Progress and transition capture use a local callback count for each stage, so wrappers with offset step numbers do not shift the transition. Each stage must still emit one callback per Euler evaluation; missing or extra callbacks produce an explicit error.

Timing does not force CUDA synchronization by default. For synchronized diagnostic measurements, set `SELFLIFT_TIMING_SYNC=1` before starting ComfyUI. This synchronizes the model's CUDA device at timing boundaries and can reduce execution overlap; leave it unset for normal use. CPU execution never invokes CUDA synchronization.

Set `SELFLIFT_MEMORY_LOG=1` before starting ComfyUI to enable `[SelfLift memory]` snapshots at sampling/transition boundaries and around upscaler loading/inference. They report process RSS, system available RAM and, for initialized CUDA devices, PyTorch allocated/reserved memory, allocator peak and device used/free memory in MiB. Device memory includes other allocations and processes; `process_peak_allocated` is the allocator peak since its last reset, **not a per-stage peak**. Boundary snapshots can miss temporary peaks. The logger never resets shared statistics, forces CUDA synchronization or changes model residency. CPU execution reports host memory only. Enabled telemetry adds some overhead, so compare runs with matching logging settings.

Intermediate PNG dumps require `SELFLIFT_DEBUG=1` before starting ComfyUI. The node creates `debug/` automatically; an existing directory alone no longer enables decoding. Debug output can decode entire intermediate videos to save their first frames, substantially increasing time and memory even when `rho=0`. Leave this option unset for normal use. Debug files are ignored by Git.

The transition finishes the audio boundary update early and releases obsolete low-resolution states before lifting. Decoded pixel-anchor frames are released before VAE encoding. Model residency remains controlled by ComfyUI; these changes release ordinary tensors without forcing model unloads or clearing the CUDA cache.

## Applicability

`latent_image` must be an all-zero size template, including any audio streams. Encoded/init latents and `noise_mask` are rejected before sampling; img2img and inpainting through this input are not supported. H3 keyframes and references supplied through conditioning remain supported.

`sigmas` must be a one-dimensional floating-point tensor with finite, nonnegative, non-increasing values. Only the final sigma may be zero, and the high-resolution starting sigma (`sigmas[transition_step]`) must be less than 1. Empty and single-value schedules return the input unchanged. Active schedules need at least two steps; `lowres_scale` must be between 0.25 and 1. A nonzero final sigma retains the sampler's partial-denoising behavior. `lowres_scale=1` still performs the transition and re-noising; use a native sampler for a full-resolution baseline.

Use the VAE belonging to the sampled model so the pixel anchor remains in the same latent space. The paper requires the backbone to support both selected resolutions. Its preliminary Wan2.1 video experiment found that unsupported token sequence lengths destabilized structure, so H3 resolutions, temporal behavior, and quality must be validated independently. H3's standard 768-pixel short edge becomes 384 pixels at `lowres_scale=0.5`, which may be outside the backbone's training distribution even when the transition itself is correct. Results from the older probe-based implementation are invalid for the current NFE-equivalent path.

### Diagnosing H3

Use the same prompt and seed with `transition_step=6` and `lowres_scale=0.5`:

| Test | `upscaler_model` | `rho` | weights | Meaning |
| --- | --- | ---: | --- | --- |
| Direct route | `none` | 0 | any | Nearest-neighbor latent lift only |
| Pixel route | `none` | 1 | `1 / 1` | Pure H3 VAE pixel anchor |
| Paper-like SelfLift-zero | `none` | 0.3 | `0.5 / 1` | Paper's image parameters with an experimental video risk map |
| Strong H3 SelfLift-zero | `none` | 0.6 | `1 / 1` | Tested H3 starting point; reduce only if overly smooth |
| External lifter | H3 checkpoint | 0 | any | Learned H3 lift, not SelfLift-zero |

If the pure pixel route is clean but partial correction is not, H3 needs a different spatiotemporal risk/mask rule or different correction parameters. If the pure pixel route is also corrupted, the H3 transition endpoint/VAE round trip does not provide the stable complementary anchor required by SelfLift-zero. If only the external lifter is clean, the practical H3 path is progressive sampling with that learned lifter; it should not be described as SelfLift-zero.

A post-fix controlled H3 run (one reference prompt/seed, 5 seconds, 8-step simple Euler) found a clean native baseline and a clean pure pixel anchor. The nearest route produced widespread outline/oil-paint artifacts; the paper's image setting (`rho=0.3`, `w_min=0.5`) left most of them visible. Raising `rho` to 0.6 improved the result, while `rho=0.6` with `w_min=w_max=1` and `rho=1` with `w_min=0.5,w_max=1` removed the main artifacts. This single-seed result does not establish general H3 compatibility or universal parameters. It does show that SelfLift-zero can operate on H3 in at least this configuration: in this run, the H3 direct-lift error was broader than on the paper's image backbones, so their sparse correction parameters did not transfer. For an H3 SelfLift-zero trial, start with `upscaler_model=none`, `rho=0.6`, and `w_min=w_max=1`, then reduce the correction only if the result is overly smooth. The external learned lifter remains the practical default and is not SelfLift-zero.

## Not included

SelfLift-rich (the distilled latent lifter + On-Policy Self Recovery) requires training and is outside this plugin.

## References and acknowledgements

- SelfLift paper: [SelfLift: Accelerating Few-Step Diffusion via Self-Recovering Resolution Transition](https://arxiv.org/abs/2609.02036)
- Optional MiniMax H3 latent upscaler checkpoint and download: [LBH-123-AI/Minimax_h3_latent_Upscaler](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler)
- Original ComfyUI integration and inference implementation: [LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler)

Thanks to LBH-123-AI for publishing the MiniMax H3 latent upscaler weights and ComfyUI implementation. They made the optional learned H3 lifting path in this plugin possible. This external lifter remains separate from the SelfLift paper's SelfLift-rich model.

```
@article{wen2026selflift,
  title={SelfLift: Accelerating Few-Step Diffusion via Self-Recovering Resolution Transition},
  author={Wen, Tingyan et al.},
  journal={arXiv:2609.02036},
  year={2026}
}
```
