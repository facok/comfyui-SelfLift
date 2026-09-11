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

- **SelfLift Progressive Sampler (MiniMax H3)** (`selflift`): H3 AV latents (e.g. from *Empty MiniMax H3 AV Latent*). The audio stream has no spatial dimensions and continues the reused Euler boundary step without spatial lifting; keyframe condition latents are rescaled to the low-res grid for the prefix. This is an engineering extension, not a configuration validated by the paper.
- **SelfLift Progressive Sampler (Image)** (`selflift`): 4D image latents (e.g. *Empty Latent Image*).
- **H3 Temporal State Transport (TST)** (`selflift`): `MODEL` → `MODEL` patch node applying training-free Temporal State Transport correction (TST, [arXiv:2609.08505](https://arxiv.org/abs/2609.08505)) to H3's joint packed attention. It works with any standard sampler node (KSampler/SamplerCustom), not only the SelfLift sampler. See the dedicated section below.

### Parameters

- `transition_step`: number of denoiser evaluations executed at low resolution. This selects the paper's transition boundary `t_r`; it is a step count, not the literal continuous-time value. Valid values are `1` through `N-1` for an `N`-step schedule. Use `3` for the paper's 4-step FLUX.2-Klein setting and `6` for its 8-step Z-Image-Turbo setting.
- `lowres_scale`: spatial scale of the prefix (paper: 0.5 = ¼ tokens).
- `rho`: fraction of the highest-risk locations corrected toward the pixel-VAE anchor. `rho=0` skips the pixel route and all artifact-aware correction.
- `w_min`, `w_max`: correction range inside the selected mask. Values must satisfy `0 <= w_min <= w_max <= 1`. The paper uses `0.5 / 1.0`; `rho=1` with `w_min=w_max=1` is the pure pixel anchor. Setting both weights to zero skips the pixel route and correction, regardless of `rho`.
- `latent_upsample` (image node only): direct-lift interpolation. The paper uses `nearest`; `bilinear` is an optional experiment.
- `upscaler_model` (H3 node only): learned 3D-conv direct lifter. `none` uses nearest-neighbor lifting. Using an external model with `rho>0` is a hybrid experiment, not paper-defined SelfLift-zero.
- `highres_tiling` (H3 node only, default off): experimental automatic tiling. High-resolution preparation evaluates 1–8 tiles and selects the first estimated fit. Capacity includes free memory plus ComfyUI-managed resident weights on the same device (deduplicated), capped at device capacity. It subtracts a weight allowance using ComfyUI's minimum weight-memory ratio, capped at the sampling model's size, and ComfyUI's minimum inference reserve. This avoids treating reclaimable weights as permanently unavailable; actual residency remains under ComfyUI's control. The estimate includes conditions and full-state buffers. The axis follows the longer 2×2-patch dimension; cores contain at least two patches. Tiny latents use one tile. If no candidate fits, the largest permitted count is used and logged with `estimate_fits=false`; avoiding OOM is not guaranteed. Video predictions use normalized overlap blending before the regular Euler update. Audio inputs and R2V references remain complete, but only the first tile's audio prediction is retained. Other tiles still compute audio; improved audio quality is not established. Keyframes are cropped and global positions preserved. Cross-tile attention is absent, so quality and speed may change. ControlNet is unsupported. Low-resolution sampling and the transition are unchanged.
- `seed`, `cfg`, `sampler`, and `sigmas` follow `SamplerCustom` semantics. Only standard Euler with `s_churn=0` is accepted.

| Node | `transition_step` | `lowres_scale` | `rho` | `w_min / w_max` | Direct lift |
| --- | ---: | ---: | ---: | ---: | --- |
| MiniMax H3 | 6 | 0.5 | 0 | `0.5 / 1.0` | First installed H3 upscaler, otherwise nearest |
| Image | 6 | 0.5 | 0.3 | `0.5 / 1.0` | Nearest |

The image defaults match the paper's 8-step Z-Image-Turbo setup. For the 4-step FLUX.2-Klein setup, change `transition_step` to `3` and `rho` to `0.4`.

The transition does not add a denoiser evaluation. The final low-resolution Euler evaluation supplies Eq. 3; after lifting and re-noising, its prediction also completes that Euler interval. With tiling off, a schedule with `N` steps therefore remains exactly `N` NFEs: `transition_step` at low resolution and the rest at target resolution. Tiling preserves the number of Euler updates and progress callbacks but doubles the spatial model forwards per high-resolution evaluation when two tiles are used (before accounting for CFG). SelfLift-zero adds one VAE decode → resize → encode round trip unless `rho=0` or both correction weights are zero. With an H3 checkpoint installed, the H3 defaults take the latent-only external path. Select `upscaler_model=none` and set `rho>0` with nonzero correction weights to run SelfLift-zero.

## H3 Temporal State Transport (TST)

An experimental port of [TST](https://github.com/lytang63/temporal-state-transport) to MiniMax H3. The paper evaluates Wan2.2, not H3; this node is an engineering adaptation, not a validated configuration. TST diagnoses temporal attention instead of blindly strengthening it: fragmented transport (attention mass concentrated on too few frames) drifts details, while over-mixing (mass spread too uniformly) breaks motion physics.

H3 has no separate temporal attention — it is a single-stream transformer over packed `[text | cond/refs | audio | video]` tokens — so the node builds the frame-level transport operator `A` (F×F) per head from spatial mean-pooled post-RoPE queries/keys of the video segment (always the last packed segment), computes Spectral Tension `T = H_row - H_vN` (paper Eqs. 1–3), and applies the homeostatic query temperature `γ = exp(τ_eff · T)` to video-row queries only (paper Eqs. 5–6). Positive tension sharpens, negative softens. `τ_eff` follows the paper's cosine schedules: stronger in deeper layers and earlier denoising steps. Text, audio and reference rows are never rescaled. The intervention runs inside ComfyUI's `optimized_attention_override` hook, so it is backend-agnostic (sage/flash/SDPA/kitchen). The whole process is training-free, touches no model weights and has no learnable parameters; the only added work is a few small F×F matrix operations per attention call (measured 0.1–0.3 s per model forward, roughly 1–2% of sampling time).

- `tau`: correction strength; `0.2` is the paper setting. `0` disables correction while keeping the diagnostic active.
- `log_diagnostics`: logs one `[H3 TST]` line per model forward with the step, latent frame count, per-frame grid rows, mean absolute tension `|T|`, mean `γ`, and the fraction of heads with `|γ-1| > 0.03`. Per-head tension is measured before each call's own correction; cross-step movement reflects corrections from earlier calls and steps.

Frame count and per-frame grid rows are captured per forward, so the low- and high-resolution phases of the SelfLift sampler are each handled correctly. Layer indices come from actual video-attention call counts; step indices match the current sigma against `sample_sigmas`, so no call-count assumptions break under CFG or phase splits.

A single-workflow, single-seed sweep (Ref2VA, 5 s, 9-step Euler, CFG 1) found net **positive** tension on this content (the over-mixing direction, opposite to the fragmentation dominance reported for Wan), baseline `|T|` rising from 0.449 at the first step to 0.486 at the last, and a dose-response curve with the measured `|T|` minimum at `tau=0.2`. At `tau=0.5`, `|T|` rose above baseline and on-screen text/details visibly degraded, matching the paper's warning about oversized `tau`. This is one seed on one prompt; treat `0.2` as a starting point and re-validate per content.

Limitations: TST is **not compatible with `highres_tiling`** — wrappers run in registration order, so the node only sees the full-resolution shape outside the tiling wrapper; tile attention calls fail the length guard and TST is skipped with a console warning. The pooled-prototype operator is an approximation of the exact video→video attention mass, and scaling video queries also shifts their attention to text/audio columns (the paper's Wan target has separate temporal attention, so this leakage does not exist there). Calibration on real H3 runs (set `SELFLIFT_TST_EXACT=1` before starting ComfyUI to probe the exact per-position frame-mass operator on one mid-layer call per forward, logged as `[H3 TST exact]`) shows 89–100% per-head sign agreement between prototype and exact tension and prototype `|T|` within roughly 10% of exact, with a slight overestimation bias — the correction direction is reliable, the magnitude is approximate.

## Timing and transition memory

With H3 high-resolution tiling enabled, sampling preparation supplies a memory-estimation shape based on the largest padded tile, retaining the complete audio and reference conditions. Text and cropped keyframe tokens are included, with an allowance of eight full-size FP32 latent buffers for sampling and blending. This uses ComfyUI's existing H3 memory heuristic; it is not a measured peak or a hard limit. The actual latent, conditioning and sigma schedule remain full-size. ComfyUI still controls loading, residency, offloading and additional-model reserves. `[SelfLift tiling memory]` reports the tile size and estimated minimum/preferred workspace before those reserves. Tiny latents keep the original preparation path.

`[SelfLift plan]` records the actual low/target latent shapes after rounding, spatial lift ratios, low/high NFE counts, transition prediction/resume sigmas, CFG and enabled lift routes. These are latent dimensions, not decoded pixel dimensions. `[SelfLift upscaler]` records the checkpoint, dtype, scale embedding, chunk/overlap, number of windows, largest actual input window and conservative workspace budget. Window lengths include padding and overlap and are measured in latent time positions, not video frames. `estimated_workspace` is a heuristic passed to ComfyUI, not measured peak memory or a hard cap; model weights are accounted for separately by the manager.

Both nodes log `[SelfLift timing]` messages for low-resolution sampling, the transition (endpoint preparation, paired lifts, correction/debug output, and re-noising), and high-resolution sampling. Sampling logs include each step and the stage total. The first step includes sampler/model preparation; callback intervals include previews and the preceding Euler update. The stage total also includes final updates, cleanup, and, for the high-resolution stage, output transfer. These are wall-clock measurements, not isolated GPU kernel timings.

Progress and transition capture use a local callback count for each stage, so wrappers with offset step numbers do not shift the transition. Each stage must still emit one callback per Euler evaluation; missing or extra callbacks produce an explicit error.

Timing does not force CUDA synchronization by default. For synchronized diagnostic measurements, set `SELFLIFT_TIMING_SYNC=1` before starting ComfyUI. This synchronizes the model's CUDA device at timing boundaries and can reduce execution overlap; leave it unset for normal use. CPU execution never invokes CUDA synchronization.

Set `SELFLIFT_MEMORY_LOG=1` before starting ComfyUI to enable `[SelfLift memory]` snapshots at sampling/transition boundaries and around upscaler loading/inference. They report process RSS, system available RAM and, for initialized CUDA devices, PyTorch allocated/reserved memory, allocator peak and device used/free memory in MiB. Device memory includes other allocations and processes; `process_peak_allocated` is the allocator peak since its last reset, **not a per-stage peak**. Boundary snapshots can miss temporary peaks. The logger never resets shared statistics, forces CUDA synchronization or changes model residency. CPU execution reports host memory only. Enabled telemetry adds some overhead, so compare runs with matching logging settings.

Intermediate PNG dumps require `SELFLIFT_DEBUG=1` before starting ComfyUI. The node creates `debug/` automatically; an existing directory alone no longer enables decoding. Debug output can decode entire intermediate videos to save their first frames, substantially increasing time and memory even when `rho=0`. Leave this option unset for normal use. Debug files are ignored by Git.

The transition finishes the audio boundary update early and releases obsolete low-resolution states before lifting. Decoded pixel-anchor frames are released before VAE encoding. Model residency remains controlled by ComfyUI; these changes release ordinary tensors without forcing model unloads or clearing the CUDA cache.

## Applicability

`latent_image` defines the target size and may contain an existing initialization latent, including audio streams. `noise_mask` is still rejected; H3 keyframes and references supplied through conditioning remain supported.

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
- TST paper and code: [Temporal State Transport in Video Generation](https://arxiv.org/abs/2609.08505), [lytang63/temporal-state-transport](https://github.com/lytang63/temporal-state-transport)
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
