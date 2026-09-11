# comfyui-SelfLift

[中文说明](README_CN.md)

Progressive-resolution sampling for ComfyUI: run the early denoising steps at low resolution, lift the result to full resolution, and finish there — faster generation with no training. Based on the [SelfLift paper](https://arxiv.org/abs/2609.02036) (SelfLift-zero) for rectified-flow image models, plus an experimental MiniMax H3 audio-video adaptation that the paper does not validate. Also included: an experimental H3 port of [TST](https://arxiv.org/abs/2609.08505) temporal-attention correction.

All nodes are in the `selflift` category. The samplers use `sampler`/`sigmas` inputs like `SamplerCustom`; connect the standard `euler` sampler from `KSamplerSelect` and the model's normal scheduler (other samplers are rejected).

## SelfLift Progressive Sampler (Image)

The paper's SelfLift-zero for rectified-flow image backbones. Connect 4D image latents (e.g. *Empty Latent Image*) and the model's own VAE.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `transition_step` | 6 | Steps run at low resolution. Paper: 6 of 8 for Z-Image-Turbo, 3 of 4 for FLUX.2-Klein |
| `lowres_scale` | 0.5 | Spatial scale of the low-res prefix |
| `rho` | 0.3 | Fraction of high-risk locations corrected toward the pixel-VAE anchor. FLUX.2-Klein: 0.4. 0 disables the anchor |
| `w_min` / `w_max` | 0.5 / 1.0 | Correction strength range inside the selected locations |
| `latent_upsample` | nearest | Direct-lift interpolation; `bilinear` is an option |

The transition adds no denoiser evaluations: an N-step schedule stays exactly N NFEs, plus one VAE decode → upscale → re-encode round trip unless `rho=0`.

## SelfLift Progressive Sampler (MiniMax H3)

Experimental H3 audio-video adaptation (not paper-validated). Two modes:

- **External upscaler (default)**: an installed H3 checkpoint with `rho=0` — a learned latent-only lift. Practical default, but not SelfLift-zero.
- **SelfLift-zero**: `upscaler_model=none` with `rho>0`. Suggested H3 starting point: `rho=0.6`, `w_min=w_max=1` (see *Diagnosing H3*).

Extra inputs: `upscaler_model` and `highres_tiling` (experimental: splits the high-res phase into 1–8 spatial tiles to save VRAM; only the first tile's audio is kept, there is no cross-tile attention, ControlNet is unsupported, and quality/speed may change).

### Optional H3 upscaler

Download from [LBH-123-AI/Minimax_h3_latent_Upscaler](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler) into `ComfyUI/models/latent_upscale_models/` and restart. The node selects the first filename containing `h3`, otherwise `none` (nearest-neighbor lift).

## H3 Temporal State Transport (TST)

A `MODEL` → `MODEL` patch node that improves the temporal stability of H3 videos at inference time — no training, no extra model, ~1–2% runtime overhead. Typical problems it counteracts: details that flicker or morph between frames (logos, on-screen text, textures), identity drift of people and outfits, and physically implausible motion. It works with any standard sampler node, not only the SelfLift sampler. It cannot create detail the model does not have.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `tau` | 0.2 | Correction strength. 0.2 is the recommended value; 0.5 is visibly too strong. 0 disables correction but keeps diagnostics |
| `log_diagnostics` | on | Logs one `[H3 TST]` line per model forward |

How it works, in one paragraph: instead of blindly strengthening cross-frame attention, TST measures a signed "Spectral Tension" of the frame-level attention operator and corrects only imbalanced heads — sharpening over-mixed ones, softening fragmented ones — with stronger correction in deeper layers and earlier steps. Theory and derivations are in the [paper](https://arxiv.org/abs/2609.08505).

The diagnostics line reports latent frames, grid rows per frame, tension before/after the within-call correction, mean gamma, the share of corrected heads, and TST's own runtime. On real H3 runs the correction direction matched the exact attention operator for 89–100% of heads. Set `SELFLIFT_TST_EXACT=1` before starting ComfyUI to log the exact-operator comparison on one layer per forward (debug only, slow).

Limitations: MiniMax H3 models only (other models are skipped silently); **not compatible with `highres_tiling`** (skipped with a console warning).

## Diagnosing H3

Same prompt and seed, `transition_step=6`, `lowres_scale=0.5`:

| Test | `upscaler_model` | `rho` | Weights | Meaning |
| --- | --- | ---: | --- | --- |
| Direct route | `none` | 0 | any | Nearest-neighbor latent lift only |
| Pixel route | `none` | 1 | `1 / 1` | Pure H3 VAE pixel anchor |
| Paper-like SelfLift-zero | `none` | 0.3 | `0.5 / 1` | Paper's image parameters |
| Strong H3 SelfLift-zero | `none` | 0.6 | `1 / 1` | Tested H3 starting point |
| External lifter | H3 checkpoint | 0 | any | Learned H3 lift |

A controlled single-seed H3 run found: clean native baseline and clean pure pixel anchor; widespread artifacts on the nearest route that the paper's image setting (`rho=0.3`) mostly left visible; `rho=0.6` with `w_min=w_max=1` removed the main artifacts. Single-seed evidence only — H3's direct-lift error is broader than on the paper's image backbones, so start strong and reduce only if the result is overly smooth.

## Logging and environment variables

- `[SelfLift plan]` — resolved low/target latent shapes, NFE counts, transition sigmas, enabled lift routes
- `[SelfLift timing]` — wall-clock time per step and per stage (low-res, transition, high-res)
- `[SelfLift upscaler]` / `[SelfLift tiling memory]` — upscaler inference and tiling memory estimates (heuristics, not measured peaks)
- `[H3 TST]` — TST per-forward diagnostics
- `SELFLIFT_TIMING_SYNC=1` — CUDA-synchronized timing (slower; for diagnostics)
- `SELFLIFT_MEMORY_LOG=1` — host/device memory snapshots at stage boundaries
- `SELFLIFT_DEBUG=1` — dump transition intermediates as PNGs under `debug/` (slow, memory-hungry)
- `SELFLIFT_TST_EXACT=1` — TST exact-operator calibration probe (debug only)

## Notes and limitations

- Only standard Euler with `s_churn=0`.
- `noise_mask` is supported on both samplers with Set Latent Noise Mask semantics (1 = generate, 0 = keep the original content; use an initialized latent to have content worth keeping). Masks at any resolution are resized to the latent grid, and a time length of 1 is shared over all frames. The keep-region is pinned to the original latent at every step, and the artifact-aware correction is restricted to the generate region. `noise_mask` is not compatible with `highres_tiling`.
- Use the VAE belonging to the sampled model so the pixel anchor stays in the same latent space.
- H3's 768-pixel short edge becomes 384 px at `lowres_scale=0.5`, which may sit outside the backbone's training distribution — validate per model.
- SelfLift-rich (the distilled lifter + On-Policy Self Recovery) requires training and is not included.

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
