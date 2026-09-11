"""SelfLift progressive-resolution sampler nodes.

Run the first sampling steps on a spatially downscaled latent, lift the clean
endpoint to the target resolution with the training-free Artifact-Aware
Consistency Lift (arXiv:2609.02036), re-noise it at the transition sigma, and
finish the schedule at full resolution.

Two front-ends share the same engine:
- SelfLiftH3Sampler: MiniMax H3 audio-video (nested AV latents; the audio stream
  has no spatial dimensions and continues through the reused Euler boundary step).
  This is an experimental extension beyond the paper's image-model evaluation.
- SelfLiftImageSampler: compatible rectified-flow image models.
"""

import logging
import os
import time

import torch

import comfy.k_diffusion.sampling
import comfy.model_management
import comfy.model_patcher
import comfy.model_sampling
import comfy.nested_tensor
import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview

from . import selflift
from . import h3_upscaler
from . import h3_tiling
from . import h3_tst
from .diagnostics import log_memory


class _StageTimer:
    def __init__(self, stage, device, resolution=None, pixel_scale=1):
        self.stage = stage
        self.resolution = resolution
        self.pixel_scale = pixel_scale
        self.device = torch.device(device)
        self.synchronize = os.environ.get("SELFLIFT_TIMING_SYNC", "0") == "1" and self.device.type == "cuda"
        self.started = self.previous = self._now()
        timing_mode = "CUDA-synchronized wall time" if self.synchronize else "wall time; no forced CUDA sync"
        suffix = ""
        if resolution is not None:
            pixels = tuple(int(value * self.pixel_scale) for value in resolution)
            suffix = f" latent={resolution} pixels={pixels}"
        logging.info("[SelfLift timing] %s start (%s)%s", stage, timing_mode, suffix)

    def _now(self):
        if self.synchronize:
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def mark(self, label):
        current = self._now()
        logging.info("[SelfLift timing] %s %s: %.3fs", self.stage, label, current - self.previous)
        self.previous = current

    def finish(self):
        current = self._now()
        logging.info("[SelfLift timing] %s total: %.3fs; after last mark: %.3fs",
                     self.stage, current - self.started, current - self.previous)


def _upscaler_input():
    models = ["none"] + h3_upscaler.list_upscaler_models()
    h3_models = [name for name in models[1:] if "h3" in name.lower()]
    default = h3_models[0] if h3_models else "none"
    return (models, {"default": default,
                     "tooltip": "External H3 latent upscaler (models/latent_upscale_models). The first detected H3 model is selected by default; 'none' uses nearest-neighbor lifting."})


def _streams(samples):
    if samples.is_nested:
        return list(samples.unbind()), True
    return [samples], False


def _pack(streams, nested):
    if nested:
        return comfy.nested_tensor.NestedTensor(streams)
    return streams[0]


def _validate_sampling(model_sampling, sampler):
    if not isinstance(model_sampling, comfy.model_sampling.CONST):
        raise ValueError("SelfLift requires a rectified-flow model")
    if not isinstance(sampler, comfy.samplers.KSAMPLER) or sampler.sampler_function is not comfy.k_diffusion.sampling.sample_euler:
        raise ValueError("SelfLift requires the standard Euler sampler")
    if sampler.extra_options.get("s_churn", 0.0) != 0.0:
        raise ValueError("SelfLift requires Euler with s_churn=0")


def _validate_schedule(sigmas, transition_step):
    if sigmas.ndim != 1 or not sigmas.is_floating_point():
        raise ValueError("SelfLift: sigmas must be a one-dimensional floating-point tensor")
    if not torch.isfinite(sigmas).all() or (sigmas < 0).any():
        raise ValueError("SelfLift: sigmas must be finite and nonnegative")
    if sigmas.numel() < 2:
        return
    if not isinstance(transition_step, int) or not 1 <= transition_step <= sigmas.numel() - 2:
        raise ValueError(f"SelfLift: transition_step {transition_step} out of range for {sigmas.numel() - 1} steps")
    if (sigmas[1:] > sigmas[:-1]).any():
        raise ValueError("SelfLift: sigmas must be non-increasing")
    if (sigmas[:-1] <= 0).any():
        raise ValueError("SelfLift: only the final sigma may be zero")
    if sigmas[transition_step] >= 1:
        raise ValueError("SelfLift: the high-resolution starting sigma must be less than 1")


def _validate_latent_input(latent_image):
    """Validate the latent template; returns the normalized noise_mask or None.

    Accepted mask shapes: [B, H, W], [B, 1, H, W], [B, 1, T, H, W] (1 = generate,
    0 = keep the original content). Spatial dims must match the latent; a time
    length of 1 shares the mask over all frames. Normalized to [B, 1, H, W] for
    image latents and [B, 1, T|1, H, W] for video latents.
    """
    streams, _ = _streams(latent_image["samples"])
    if not streams or streams[0].ndim not in (4, 5):
        raise ValueError("SelfLift: expected a 4D image or 5D video latent size template")
    for stream in streams:
        if stream.ndim == 0 or any(size == 0 for size in stream.shape) or stream.shape[0] != streams[0].shape[0]:
            raise ValueError("SelfLift: latent streams must be nonempty and have the same batch size")
    mask = latent_image.get("noise_mask")
    if mask is None:
        return None
    video = streams[0].ndim == 5
    b, H, W = streams[0].shape[0], streams[0].shape[-2], streams[0].shape[-1]
    if mask.ndim == 3:  # [B, H, W]
        mask = mask[:, None]
    if video and mask.ndim == 4:  # [B, 1, H, W] shared over time
        mask = mask[:, :, None]
    if mask.ndim != (5 if video else 4) or mask.shape[1] != 1:
        raise ValueError("SelfLift: noise_mask must have shape [B, H, W], [B, 1, H, W], or [B, 1, T, H, W]")
    if mask.shape[0] != b or tuple(mask.shape[-2:]) != (H, W):
        raise ValueError(f"SelfLift: noise_mask shape {tuple(mask.shape)} does not match the latent batch/size ({b}, {H}, {W})")
    if video and mask.shape[2] not in (1, streams[0].shape[2]):
        raise ValueError(f"SelfLift: noise_mask time length {mask.shape[2]} does not match the latent frames {streams[0].shape[2]}")
    if not torch.isfinite(mask).all():
        raise ValueError("SelfLift: noise_mask must be finite")
    return mask.float().clamp(0.0, 1.0)


def _mask_blend_fn(anchor, mask):
    """post-CFG hook pinning x0 to the clean anchor outside the generate region.

    Exact under Euler: with the unmasked x0 pinned to the original latent each
    step, the unmasked state tracks original + the step's noise exactly.
    """
    def fn(args):
        d = args["denoised"]
        m = mask.to(device=d.device, dtype=d.dtype)
        return d * m + anchor.to(device=d.device, dtype=d.dtype) * (1.0 - m)
    return fn


def _euler_step(state, denoised, sigma, sigma_next):
    step = ((sigma_next - sigma) / sigma).to(device=state.device, dtype=state.dtype)
    return state + (state - denoised.to(state)) * step


def _resize_keyframes(cond, h, w):
    """Keyframe cond latents share the generation grid; resize them to the low-res one."""
    out = []
    for tensor, d in cond:
        kfs = d.get("minimax_keyframes")
        if kfs is None:
            out.append((tensor, d))
            continue
        d = d.copy()
        resized = []
        for kf in kfs:
            kf = dict(kf)
            lat = kf.get("latent")
            if lat is not None and (lat.shape[-2] != h or lat.shape[-1] != w):
                kf["latent"] = torch.nn.functional.interpolate(
                    lat.float(), size=(lat.shape[2], h, w), mode="trilinear", align_corners=False
                ).to(lat)
            resized.append(kf)
        d["minimax_keyframes"] = resized
        out.append((tensor, d))
    return out


def _debug_dump(vae, latents):
    """Decode transition intermediates to PNGs when SELFLIFT_DEBUG=1."""
    if os.environ.get("SELFLIFT_DEBUG", "0") != "1":
        return
    out_dir = os.path.join(os.path.dirname(__file__), "debug")
    os.makedirs(out_dir, exist_ok=True)
    logging.warning("SelfLift: debug decoding enabled; intermediate video decodes can substantially increase time and memory")
    from PIL import Image
    for name, lat in latents.items():
        if lat is None:
            continue
        img = vae.decode(lat)
        if img.ndim == 5:
            img = img.reshape(-1, img.shape[-3], img.shape[-2], img.shape[-1])
        frame = (img[0].float().cpu().numpy().clip(0.0, 1.0) * 255).round().astype("uint8")
        Image.fromarray(frame).save(os.path.join(out_dir, name + ".png"))


def progressive_sample(model, positive, negative, vae, latent_image, sampler, sigmas, seed, cfg,
                       transition_step, lowres_scale, rho, w_min, w_max, latent_upsample, latent_lifter=None,
                       highres_tiling=False):
    _validate_schedule(sigmas, transition_step)
    if sigmas.numel() < 2:
        return latent_image
    if not 0.25 <= lowres_scale <= 1.0:
        raise ValueError("SelfLift: lowres_scale must be between 0.25 and 1")
    if not 0.0 <= rho <= 1.0:
        raise ValueError("SelfLift: rho must be between 0 and 1")
    if not 0.0 <= w_min <= w_max <= 1.0:
        raise ValueError("SelfLift: weights must satisfy 0 <= w_min <= w_max <= 1")
    noise_mask = _validate_latent_input(latent_image)

    model_sampling = model.get_model_object("model_sampling")
    _validate_sampling(model_sampling, sampler)

    streams, nested = _streams(comfy.sample.fix_empty_latent_channels(
        model, latent_image["samples"], latent_image.get("downscale_ratio_spacial", None),
        latent_image.get("downscale_ratio_temporal", None)))
    high_model = h3_tiling.tiled_model(model, [tuple(stream.shape) for stream in streams]) if highres_tiling else model
    video = streams[0].ndim == 5
    if video:
        b, c, t, H, W = streams[0].shape
    else:
        b, c, H, W = streams[0].shape
        t = None
    h = max(2, round(H * lowres_scale / 2) * 2)
    w = max(2, round(W * lowres_scale / 2) * 2)
    low_shape = (b, c, t, h, w) if video else (b, c, h, w)
    logging.info("[SelfLift plan] low_latent=%s target_latent=%s spatial_lift=(%.4f, %.4f) "
                 "low_nfe=%d high_nfe=%d sigma_prediction=%.8g sigma_resume=%.8g "
                 "rho=%.4f weights=(%.4f, %.4f) direct_lift=%s pixel_anchor=%s cfg=%.4f mask=%s",
                 low_shape, tuple(streams[0].shape), H / h, W / w,
                 transition_step, sigmas.numel() - 1 - transition_step,
                 sigmas[transition_step - 1].item(), sigmas[transition_step].item(),
                 rho, w_min, w_max,
                 "skipped" if rho == 1.0 and w_min == 1.0 else "external" if latent_lifter is not None else latent_upsample,
                 rho > 0.0 and w_max > 0.0, cfg,
                 "none" if noise_mask is None else f"{tuple(noise_mask.shape)}")

    device = comfy.model_management.intermediate_device()
    source_video = streams[0].to(device)
    audio_streams = [s.to(device) for s in streams[1:]]

    def masked_model(base, video_anchor, m):
        # pin the unmasked region to the clean anchor via a post-CFG x0 blend;
        # ComfyUI's denoise_mask path is unusable here: the high-res resume
        # latent deliberately carries noise, which would contaminate its anchor
        if m is None:
            return base
        patched = base.clone()
        if nested:
            flat_mask = m.reshape(b, 1, -1).repeat(1, 1, c).to(video_anchor.device)
            flat_anchor = video_anchor.reshape(b, 1, -1)
            for s in audio_streams:
                flat_mask = torch.cat([flat_mask, torch.ones_like(s.reshape(b, 1, -1))], dim=-1)
                flat_anchor = torch.cat([flat_anchor, s.reshape(b, 1, -1)], dim=-1)
            hook = _mask_blend_fn(flat_anchor, flat_mask)
        else:
            hook = _mask_blend_fn(video_anchor, m)
        patched.model_options = comfy.model_patcher.set_model_options_post_cfg_function(patched.model_options, hook)
        return patched

    if video:
        low_video = torch.nn.functional.interpolate(
            source_video.float(), size=(t, h, w), mode="trilinear", align_corners=False
        ).to(dtype=source_video.dtype)
    else:
        low_video = torch.nn.functional.interpolate(
            source_video.float(), size=(h, w), mode="bilinear", align_corners=False
        ).to(dtype=source_video.dtype)
    m_full = m_low = None
    if noise_mask is not None:
        if video and noise_mask.shape[2] == 1 and t > 1:
            m_full = noise_mask.expand(b, 1, t, H, W)
        else:
            m_full = noise_mask
        if video:
            m_low = torch.nn.functional.interpolate(
                m_full.reshape(b * t, 1, H, W).float(), size=(h, w), mode="bilinear", align_corners=False
            ).reshape(b, 1, t, h, w)
        else:
            m_low = torch.nn.functional.interpolate(
                m_full.float(), size=(h, w), mode="bilinear", align_corners=False)
        m_low = m_low.clamp(0.0, 1.0)
    low_anchor = low_video
    video_anchor = source_video
    low_latent = _pack([low_video] + audio_streams, nested)
    low_model = masked_model(model, low_anchor, m_low)
    high_model = masked_model(high_model, video_anchor, m_full)
    del source_video, low_video
    del streams
    noise_low = comfy.sample.prepare_noise(low_latent, seed, latent_image.get("batch_index", None))

    total_steps = sigmas.shape[-1] - 1
    callback = latent_preview.prepare_callback(model, total_steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    if video:  # keyframe cond latents share the generation grid on MiniMax H3
        positive_low = _resize_keyframes(positive, h, w)
        negative_low = _resize_keyframes(negative, h, w)
    else:
        positive_low, negative_low = positive, negative

    transition = {}
    low_evaluations = 0

    def callback_low(step, x0, x, total):
        nonlocal low_evaluations
        step = low_evaluations
        low_evaluations += 1
        if low_evaluations > transition_step:
            raise RuntimeError("SelfLift: too many low-resolution callbacks for the Euler schedule")
        if step == transition_step - 1:
            transition["state"] = x
            transition["x0"] = x0
        # Preview decoders expect the target grid.  Decode a temporary lifted
        # preview while keeping the sampler state at its true low resolution.
        preview_x0 = x0
        if video:
            preview_streams, preview_nested = _streams(x0)
            preview_video = torch.nn.functional.interpolate(
                preview_streams[0].float(), size=(t, H, W), mode="trilinear", align_corners=False
            ).to(preview_streams[0].dtype)
            preview_x0 = _pack([preview_video] + preview_streams[1:], preview_nested)
            del preview_video, preview_streams
        elif x0.ndim == 4:
            preview_x0 = torch.nn.functional.interpolate(
                x0.float(), size=(H, W), mode="bilinear", align_corners=False
            ).to(x0.dtype)
        result = callback(step, preview_x0, preview_x0, total_steps)
        del preview_x0
        low_timer.mark(f"step {step + 1}/{transition_step}" + (" (includes setup)" if step == 0 else ""))
        return result

    # The final low-resolution model evaluation is the Eq. 3 prediction. Its Euler
    # update is discarded and rebuilt after the resolution transition, preserving NFE.
    resolution_scale = (getattr(model.get_model_object("latent_format"), "spacial_downscale_ratio", 1)
                        if video else getattr(model.get_model_object("latent_format"), "spacial_downscale_ratio", 1))
    low_timer = _StageTimer("low_resolution", low_model.load_device, (h, w), resolution_scale)
    log_memory("low_resolution start", low_model.load_device)
    comfy.samplers.sample(low_model, noise_low, positive_low, negative_low, cfg, low_model.load_device,
                          sampler, sigmas[:transition_step + 1], low_model.model_options,
                          latent_image=low_latent, callback=callback_low,
                          disable_pbar=disable_pbar, seed=seed)
    if low_evaluations != transition_step:
        raise RuntimeError(f"SelfLift: expected {transition_step} low-resolution callbacks, received {low_evaluations}; check sampler wrappers")
    low_timer.finish()
    log_memory("low_resolution end", model.load_device)
    transition_timer = _StageTimer("transition", model.load_device, (H, W), resolution_scale)
    low_streams, nested = _streams(transition.pop("state"))
    x0_streams, _ = _streams(transition.pop("x0"))
    sigma_k = sigmas[transition_step - 1]
    sigma_next = sigmas[transition_step]
    auxiliary_next = [_euler_step(state.to(device), denoised.to(device), sigma_k, sigma_next)
                      for state, denoised in zip(low_streams[1:], x0_streams[1:])]
    latent_format = model.get_model_object("latent_format")
    z0_low_vae = latent_format.process_out(x0_streams[0].float()).to(device)
    del low_latent, noise_low, low_streams, x0_streams, positive_low, negative_low
    transition_timer.mark("prepare_endpoint")
    log_memory("transition endpoint_ready", model.load_device)

    # Artifact-Aware Consistency Lift (Eqs. 4-9); skip branches the weights discard
    need_pix = rho > 0.0 and w_max > 0.0
    need_lat = not (rho >= 1.0 and w_min >= 1.0 and w_max >= 1.0)
    z_lat_vae, z_pix_vae = selflift.paired_lifts(
        z0_low_vae, vae, (H, W), latent_upsample, latent_lifter,
        need_lat=need_lat, need_pix=need_pix)
    transition_timer.mark("paired_lifts")
    log_memory("transition lifts_ready", model.load_device)
    z_lat = latent_format.process_in(z_lat_vae) if z_lat_vae is not None else None
    z_pix = latent_format.process_in(z_pix_vae) if z_pix_vae is not None else None
    z0_high = selflift.artifact_aware_consistency_lift(z_lat, z_pix, rho, w_min, w_max, mask=m_full)
    if m_full is not None:
        # restore the keep-region with the true original instead of any lifted estimate
        m_cast = m_full.to(device=z0_high.device, dtype=z0_high.dtype)
        z0_high = z0_high * m_cast + video_anchor * (1.0 - m_cast)
    if os.environ.get("SELFLIFT_DEBUG", "0") == "1":
        _debug_dump(vae, {
            "z0_low": z0_low_vae,
            "z_lat": z_lat_vae,
            "z_pix": z_pix_vae,
            "z0_high": latent_format.process_out(z0_high),
        })
    z0_high = z0_high.to(device)
    del z0_low_vae, z_lat_vae, z_pix_vae, z_lat, z_pix
    transition_timer.mark("correction_and_debug")
    log_memory("transition correction_ready", model.load_device)

    # Re-noise the corrected video/image endpoint at the sigma of the reused model
    # evaluation, then complete that Euler interval without another denoiser call.
    video_noise = comfy.sample.prepare_noise(z0_high, (seed + 1) % (1 << 64),
                                             latent_image.get("batch_index", None)).to(z0_high)
    video_state = model_sampling.noise_scaling(sigma_k, video_noise, z0_high)
    next_streams = [_euler_step(video_state, z0_high, sigma_k, sigma_next)] + auxiliary_next
    del z0_high, video_noise, video_state, auxiliary_next
    resume_streams = [model_sampling.inverse_noise_scaling(sigma_next, s) for s in next_streams]
    resume_latent = model.model.process_latent_out(_pack(resume_streams, nested))
    resume_noise = _pack([torch.zeros_like(s) for s in resume_streams], nested)
    del next_streams, resume_streams
    transition_timer.mark("renoise")
    transition_timer.finish()
    log_memory("transition end / high_resolution start", model.load_device)

    high_evaluations = 0

    def callback_high(step, x0, x, total):
        nonlocal high_evaluations
        step = high_evaluations
        high_evaluations += 1
        if high_evaluations > total_steps - transition_step:
            raise RuntimeError("SelfLift: too many high-resolution callbacks for the Euler schedule")
        result = callback(step + transition_step, x0, x, total_steps)
        high_timer.mark(f"step {step + 1}/{total_steps - transition_step}" + (" (includes setup)" if step == 0 else ""))
        return result

    high_timer = _StageTimer("high_resolution", model.load_device, (H, W), resolution_scale)
    if highres_tiling:
        logging.info("[SelfLift plan] automatic high-resolution tiling enabled; preparation selects the tile count")
    out = comfy.samplers.sample(high_model, resume_noise, positive, negative, cfg, model.load_device,
                                sampler, sigmas[transition_step:], high_model.model_options,
                                latent_image=resume_latent, callback=callback_high,
                                disable_pbar=disable_pbar, seed=seed)
    del resume_latent, resume_noise
    if high_evaluations != total_steps - transition_step:
        raise RuntimeError(f"SelfLift: expected {total_steps - transition_step} high-resolution callbacks, received {high_evaluations}; check sampler wrappers")

    result = latent_image.copy()
    result["samples"] = out.to(device=comfy.model_management.intermediate_device(),
                                dtype=comfy.model_management.intermediate_dtype())
    high_timer.finish()
    log_memory("high_resolution end", model.load_device)
    return result


class SelfLiftH3Sampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
            "vae": ("VAE", {"tooltip": "Video VAE used for the pixel re-encode anchor at the resolution transition."}),
            "latent_image": ("LATENT", {"tooltip": "Target-resolution H3 AV latent defining size and duration. Existing latent content is accepted; noise masks are not supported."}),
            "sampler": ("SAMPLER", {"tooltip": "Standard Euler only; SelfLift reuses its transition-step prediction to keep the original NFE count."}),
            "sigmas": ("SIGMAS",),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
            "cfg": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
            "transition_step": ("INT", {"default": 6, "min": 1, "max": 10000, "tooltip": "Number of low-resolution denoiser evaluations. The paper uses 6 of 8 NFEs for its 8-step image model; H3 requires independent validation."}),
            "lowres_scale": ("FLOAT", {"default": 0.5, "min": 0.25, "max": 1.0, "step": 0.05, "tooltip": "Spatial scale of the low-resolution prefix (paper: 0.5)."}),
            "rho": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Fraction of highest-risk spatiotemporal locations corrected toward the pixel-VAE anchor. The H3 default 0 uses only the external latent upscaler and skips the VAE round trip. For SelfLift-zero with upscaler_model=none, start near 0.6."}),
            "w_min": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Correction-strength floor. H3's widespread nearest-lift error can require 1.0; 0.5 is the paper's image-model setting."}),
            "w_max": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Correction-strength ceiling. Keep at 1.0 for the H3 SelfLift-zero diagnostic."}),
            "upscaler_model": _upscaler_input(),
        }, "optional": {
            "highres_tiling": ("BOOLEAN", {"default": False, "label_on": "高分辨率分块：开启", "label_off": "高分辨率分块：关闭", "tooltip": "Experimental: select 1–8 spatial tiles from available memory at high-resolution preparation. Audio input and references remain complete; only the first tile's audio prediction is retained. Quality and speed may change."}),
        }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "selflift"

    def sample(self, model, positive, negative, vae, latent_image, sampler, sigmas, seed, cfg,
               transition_step, lowres_scale, rho, w_min, w_max, upscaler_model, highres_tiling=False):
        lifter = None
        if upscaler_model != "none":
            if rho > 0.0 and w_max > 0.0:
                logging.warning("SelfLift H3: rho > 0 with an external upscaler is a hybrid experiment; select upscaler_model=none to test the paper's SelfLift-zero direct route")
            lifter = lambda z, hw: h3_upscaler.learned_latent_lift(z, hw, upscaler_model)
        return (progressive_sample(model, positive, negative, vae, latent_image, sampler, sigmas, seed, cfg,
                                   transition_step, lowres_scale, rho, w_min, w_max, "nearest",
                                   latent_lifter=lifter, highres_tiling=highres_tiling),)


class SelfLiftImageSampler:
    """SelfLift-zero for compatible rectified-flow image backbones."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
            "vae": ("VAE", {"tooltip": "VAE used for the pixel re-encode anchor at the resolution transition."}),
            "latent_image": ("LATENT", {"tooltip": "Target-resolution latent defining the output size. Existing latent content is accepted; noise masks are not supported."}),
            "sampler": ("SAMPLER", {"tooltip": "Standard Euler only; SelfLift reuses its transition-step prediction to keep the original NFE count."}),
            "sigmas": ("SIGMAS",),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
            "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
            "transition_step": ("INT", {"default": 6, "min": 1, "max": 10000, "tooltip": "Number of low-resolution denoiser evaluations. Paper: 3 of 4 for FLUX.2-Klein, 6 of 8 for Z-Image-Turbo."}),
            "lowres_scale": ("FLOAT", {"default": 0.5, "min": 0.25, "max": 1.0, "step": 0.05, "tooltip": "Spatial scale of the low-resolution prefix (paper: 0.5)."}),
            "rho": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Fraction of locations corrected toward the pixel-VAE anchor. Paper: 0.4 for FLUX.2-Klein, 0.3 for Z-Image-Turbo."}),
            "w_min": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
            "w_max": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            "latent_upsample": (["nearest", "bilinear"], {"default": "nearest", "tooltip": "Interpolation for the direct latent lift (paper: nearest)."}),
        }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "selflift"

    def sample(self, model, positive, negative, vae, latent_image, sampler, sigmas, seed, cfg,
               transition_step, lowres_scale, rho, w_min, w_max, latent_upsample):
        return (progressive_sample(model, positive, negative, vae, latent_image, sampler, sigmas, seed, cfg,
                                   transition_step, lowres_scale, rho, w_min, w_max, latent_upsample),)


class SelfLiftH3TST:
    """Training-free Temporal State Transport correction (arXiv:2609.08505) for MiniMax H3."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "tau": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05,
                              "tooltip": "Homeostatic correction strength; 0.2 is the paper setting. 0 disables correction while keeping the diagnostic active."}),
            "log_diagnostics": ("BOOLEAN", {"default": True, "label_on": "诊断日志：开启", "label_off": "诊断日志：关闭",
                                            "tooltip": "Log per-forward Spectral Tension and correction statistics to the console. Note: TST is skipped with a warning when highres_tiling is enabled on the SelfLift sampler."}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "selflift"

    def patch(self, model, tau, log_diagnostics):
        return (h3_tst.patch_model(model, tau, log_diagnostics),)


NODE_CLASS_MAPPINGS = {
    "SelfLiftH3Sampler": SelfLiftH3Sampler,
    "SelfLiftImageSampler": SelfLiftImageSampler,
    "SelfLiftH3TST": SelfLiftH3TST,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SelfLiftH3Sampler": "SelfLift Progressive Sampler (MiniMax H3)",
    "SelfLiftImageSampler": "SelfLift Progressive Sampler (Image)",
    "SelfLiftH3TST": "H3 Temporal State Transport (TST)",
}
