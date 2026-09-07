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

import torch

import comfy.k_diffusion.sampling
import comfy.model_management
import comfy.model_sampling
import comfy.nested_tensor
import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview

from . import selflift
from . import h3_upscaler


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
    """Decode transition intermediates to PNGs; enabled by creating a debug/ dir next to this file."""
    out_dir = os.path.join(os.path.dirname(__file__), "debug")
    if not os.path.isdir(out_dir):
        return
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
                       transition_step, lowres_scale, rho, w_min, w_max, latent_upsample, latent_lifter=None):
    if sigmas.shape[-1] < 2:
        return latent_image
    if not 1 <= transition_step <= sigmas.shape[-1] - 2:
        raise ValueError(f"SelfLift: transition_step {transition_step} out of range for {sigmas.shape[-1] - 1} steps")
    if not 0.0 <= rho <= 1.0:
        raise ValueError("SelfLift: rho must be between 0 and 1")
    if not 0.0 <= w_min <= w_max <= 1.0:
        raise ValueError("SelfLift: weights must satisfy 0 <= w_min <= w_max <= 1")

    model_sampling = model.get_model_object("model_sampling")
    _validate_sampling(model_sampling, sampler)

    streams, nested = _streams(comfy.sample.fix_empty_latent_channels(
        model, latent_image["samples"], latent_image.get("downscale_ratio_spacial", None),
        latent_image.get("downscale_ratio_temporal", None)))
    video = streams[0].ndim == 5
    if video:
        b, c, t, H, W = streams[0].shape
    else:
        b, c, H, W = streams[0].shape
        t = None
    h = max(2, round(H * lowres_scale / 2) * 2)
    w = max(2, round(W * lowres_scale / 2) * 2)
    low_shape = (b, c, t, h, w) if video else (b, c, h, w)

    device = comfy.model_management.intermediate_device()
    low_latent = _pack([torch.zeros(low_shape, device=device)] +
                       [torch.zeros_like(s) for s in streams[1:]], nested)
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

    def callback_low(step, x0, x, total):
        if step == transition_step - 1:
            transition["state"] = x
            transition["x0"] = x0
        return callback(step, x0, x, total_steps)

    # The final low-resolution model evaluation is the Eq. 3 prediction. Its Euler
    # update is discarded and rebuilt after the resolution transition, preserving NFE.
    comfy.samplers.sample(model, noise_low, positive_low, negative_low, cfg, model.load_device,
                          sampler, sigmas[:transition_step + 1], model.model_options,
                          latent_image=low_latent, callback=callback_low,
                          disable_pbar=disable_pbar, seed=seed)
    low_streams, nested = _streams(transition["state"])
    x0_streams, _ = _streams(transition["x0"])

    # Artifact-Aware Consistency Lift (Eqs. 4-9); skip branches the weights discard
    need_pix = rho > 0.0
    need_lat = not (rho >= 1.0 and w_min >= 1.0 and w_max >= 1.0)
    latent_format = model.get_model_object("latent_format")
    z0_low = x0_streams[0].float()
    z0_low_vae = latent_format.process_out(z0_low)
    z_lat_vae, z_pix_vae = selflift.paired_lifts(
        z0_low_vae, vae, (H, W), latent_upsample, latent_lifter,
        need_lat=need_lat, need_pix=need_pix)
    z_lat = latent_format.process_in(z_lat_vae) if z_lat_vae is not None else None
    z_pix = latent_format.process_in(z_pix_vae) if z_pix_vae is not None else None
    x0_streams[0] = selflift.artifact_aware_consistency_lift(z_lat, z_pix, rho, w_min, w_max)
    _debug_dump(vae, {
        "z0_low": z0_low_vae,
        "z_lat": z_lat_vae,
        "z_pix": z_pix_vae,
        "z0_high": latent_format.process_out(x0_streams[0]),
    })

    # Re-noise the corrected video/image endpoint at the sigma of the reused model
    # evaluation, then complete that Euler interval without another denoiser call.
    sigma_k = sigmas[transition_step - 1]
    sigma_next = sigmas[transition_step]
    x0_streams = [s.to(device) for s in x0_streams]
    video_noise = comfy.sample.prepare_noise(torch.zeros_like(x0_streams[0]), seed + 1,
                                             latent_image.get("batch_index", None))
    state_streams = [model_sampling.noise_scaling(sigma_k, video_noise, x0_streams[0])]
    state_streams.extend(s.to(device) for s in low_streams[1:])
    next_streams = [_euler_step(state, denoised, sigma_k, sigma_next)
                    for state, denoised in zip(state_streams, x0_streams)]
    resume_streams = [model_sampling.inverse_noise_scaling(sigma_next, s) for s in next_streams]
    resume_latent = model.model.process_latent_out(_pack(resume_streams, nested))
    resume_noise = _pack([torch.zeros_like(s) for s in resume_streams], nested)

    def callback_high(step, x0, x, total):
        return callback(step + transition_step, x0, x, total_steps)

    out = comfy.samplers.sample(model, resume_noise, positive, negative, cfg, model.load_device,
                                sampler, sigmas[transition_step:], model.model_options,
                                latent_image=resume_latent, callback=callback_high,
                                disable_pbar=disable_pbar, seed=seed)

    result = latent_image.copy()
    result["samples"] = out
    return result


class SelfLiftH3Sampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
            "vae": ("VAE", {"tooltip": "Video VAE used for the pixel re-encode anchor at the resolution transition."}),
            "latent_image": ("LATENT", {"tooltip": "Target-resolution latent (e.g. Empty MiniMax H3 AV Latent), defines the output size and duration."}),
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
        }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling/minimax"

    def sample(self, model, positive, negative, vae, latent_image, sampler, sigmas, seed, cfg,
               transition_step, lowres_scale, rho, w_min, w_max, upscaler_model):
        lifter = None
        if upscaler_model != "none":
            if rho > 0.0:
                logging.warning("SelfLift H3: rho > 0 with an external upscaler is a hybrid experiment; select upscaler_model=none to test the paper's SelfLift-zero direct route")
            lifter = lambda z, hw: h3_upscaler.learned_latent_lift(z, hw, upscaler_model)
        return (progressive_sample(model, positive, negative, vae, latent_image, sampler, sigmas, seed, cfg,
                                   transition_step, lowres_scale, rho, w_min, w_max, "nearest",
                                   latent_lifter=lifter),)


class SelfLiftImageSampler:
    """SelfLift-zero for compatible rectified-flow image backbones."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
            "vae": ("VAE", {"tooltip": "VAE used for the pixel re-encode anchor at the resolution transition."}),
            "latent_image": ("LATENT", {"tooltip": "Target-resolution latent (e.g. Empty Latent Image), defines the output size."}),
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
    CATEGORY = "sampling"

    def sample(self, model, positive, negative, vae, latent_image, sampler, sigmas, seed, cfg,
               transition_step, lowres_scale, rho, w_min, w_max, latent_upsample):
        return (progressive_sample(model, positive, negative, vae, latent_image, sampler, sigmas, seed, cfg,
                                   transition_step, lowres_scale, rho, w_min, w_max, latent_upsample),)


NODE_CLASS_MAPPINGS = {
    "SelfLiftH3Sampler": SelfLiftH3Sampler,
    "SelfLiftImageSampler": SelfLiftImageSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SelfLiftH3Sampler": "SelfLift Progressive Sampler (MiniMax H3)",
    "SelfLiftImageSampler": "SelfLift Progressive Sampler (Image)",
}
