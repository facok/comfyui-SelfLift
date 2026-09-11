"""Temporal State Transport (TST) correction for MiniMax H3 joint attention.

Inference-time port of TST (arXiv:2609.08505) to H3's single-stream packed
attention. H3 has no separate temporal attention, so per-head frame-level
transport operators A (F x F) are built from spatial mean-pooled post-RoPE
queries/keys of the video segment, which is always the last packed segment.
Spectral Tension T = H_row - H_vN diagnoses fragmented (T<0) versus over-mixed
(T>0) temporal states; the homeostatic query temperature gamma =
exp(tau_eff * T) rescales only video-row queries. tau_eff follows the paper's
cosine schedules: stronger in deeper layers and earlier denoising steps.

Anchored to ComfyUI sources:
- video segment last: comfy/ldm/minimax/model.py PackedLayout ("target audio
  then target video, always the last two segments")
- injection point: comfy/ldm/modules/attention.py wrap_attn, which checks
  transformer_options["optimized_attention_override"] in every backend (and
  take()s AttentionTensorContainers before calling the override)
- per-forward video shape: WrappersMP.DIFFUSION_MODEL wraps MiniMaxH3Model._forward

Known limitation: not compatible with SelfLift highres_tiling. Wrappers run in
registration order, so the TST wrapper sits outside the tiling wrapper and only
sees the full-resolution shape; tile attention calls fail the length guard and
TST skips with a warning instead of mislocating the tile's video rows.
"""

import logging
import math
import time
from functools import partial

import torch

import comfy.patcher_extension
from comfy.ldm.modules.attention import AttentionTensorContainer


def _unwrap(tensor):
    # H3 wraps q/k/v in AttentionTensorContainer, but wrap_attn take()s them
    # before calling this override (it has no container_function attribute),
    # so plain tensors are the common case here.
    return tensor.peek() if isinstance(tensor, AttentionTensorContainer) else tensor


def _spectral_tension(q, k, v0, frames, rows_per_frame, eps=1e-8):
    """Per-head signed Spectral Tension of the frame-level transport operator.

    q, k: [1, heads, S, head_dim] post-norm post-RoPE; video rows start at v0.
    Returns [heads] fp32; T>0 over-mixing, T<0 fragmented (paper Eqs. 1-3).
    """
    heads = q.shape[1]
    head_dim = q.shape[-1]
    qv = q[0, :, v0:].reshape(heads, frames, rows_per_frame, head_dim).mean(dim=2).float()
    kv = k[0, :, v0:].reshape(heads, frames, rows_per_frame, head_dim).mean(dim=2).float()
    a = (qv @ kv.transpose(-2, -1) * head_dim**-0.5).softmax(dim=-1).clamp(min=eps)
    log_f = math.log(frames)
    h_row = -(a * a.log()).sum(dim=-1).mean(dim=-1) / log_f
    gram = a @ a.transpose(-2, -1)
    trace = gram.diagonal(dim1=-2, dim2=-1).sum(dim=-1).clamp(min=eps)
    eig = torch.linalg.eigvalsh(gram / trace[:, None, None]).clamp(min=eps)
    eig = eig / eig.sum(dim=-1, keepdim=True).clamp(min=eps)
    h_vn = -(eig * eig.log()).sum(dim=-1) / log_f
    return h_row - h_vn


def _attention_override(state):
    def override(func, q, k, v, heads, mask=None, **kwargs):
        frames = state["frames"]
        rows = state["rows_per_frame"]
        qt = _unwrap(q)
        s = qt.shape[2]
        # video rows are the last frames*rows of the packed sequence; the guard
        # also rejects token-refiner calls (text length is far below frames*rows).
        # Only video attention calls advance the layer counter, so a refiner
        # running inside a forward cannot shift the layer schedule.
        if frames is None or frames < 2 or s <= frames * rows:
            if frames is not None and frames >= 2:
                state["guard_misses"] += 1
            return func(q, k, v, heads, mask=mask, **kwargs)
        state["calls_this_forward"] += 1
        tst_start = time.perf_counter()
        layers = state["layers"]
        layer = (state["calls_this_forward"] - 1) % layers
        total_steps = state["total_steps"]
        step = min(state["step"], total_steps - 1)
        layer_weight = 0.5 - 0.5 * math.cos(math.pi * layer / (layers - 1)) if layers > 1 else 1.0
        step_weight = 0.5 + 0.5 * math.cos(math.pi * step / (total_steps - 1)) if total_steps > 1 else 1.0
        tension = _spectral_tension(qt, _unwrap(k), s - frames * rows, frames, rows)
        gamma = torch.exp(tension * (state["tau"] * layer_weight * step_weight))
        if state["tau"] != 0.0:
            qt[0, :, s - frames * rows:].mul_(gamma.to(qt.dtype)[:, None, None])
        state["fw_tst_time"].append(time.perf_counter() - tst_start)
        if state["log_diagnostics"]:
            state["fw_abs_tension"].append(float(tension.abs().mean()))
            state["fw_gamma"].append(float(gamma.mean()))
            state["fw_active"].append(float(((gamma - 1.0).abs() > 0.03).float().mean()))
            if layer == layers - 1 and state["fw_gamma"]:
                logging.info(
                    "[H3 TST] step %d/%d frames=%d grid_rows=%d mean|T|=%.4f mean_gamma=%.4f active_heads=%.0f%% tst_time=%.1fms",
                    state["step"], total_steps, frames, rows,
                    sum(state["fw_abs_tension"]) / len(state["fw_abs_tension"]),
                    sum(state["fw_gamma"]) / len(state["fw_gamma"]),
                    100.0 * sum(state["fw_active"]) / len(state["fw_active"]),
                    1000.0 * sum(state["fw_tst_time"]))
                state["fw_abs_tension"].clear()
                state["fw_gamma"].clear()
                state["fw_active"].clear()
                state["fw_tst_time"].clear()
        return func(q, k, v, heads, mask=mask, **kwargs)
    return override


def _forward_wrapper(state, executor, x, timestep, context, transformer_options, minimax_payload=None, **kwargs):
    if state["calls_this_forward"] > 0:
        state["layers"] = state["calls_this_forward"]
    elif state["guard_misses"] > 0 and not state["warned_inactive"]:
        # every attention call of the previous forward was shorter than the
        # registered video segment: the wrapper sits outside a spatial-tiling
        # wrapper (wrappers run in registration order) and only saw the
        # full-resolution shape, so TST cannot locate the tile's video rows
        logging.warning("[H3 TST] inactive: packed sequence shorter than the expected video segment; "
                        "TST is not compatible with highres_tiling and was skipped")
        state["warned_inactive"] = True
    state["calls_this_forward"] = 0
    state["guard_misses"] = 0
    # diagnostics off leaves fw_tst_time unreported; drop it per forward instead of leaking floats
    state["fw_tst_time"].clear()
    video = x[0] if isinstance(x, (list, tuple)) else x
    if isinstance(video, torch.Tensor) and video.ndim == 5:
        state["frames"] = int(video.shape[2])
        state["rows_per_frame"] = ((video.shape[3] + 1) // 2) * ((video.shape[4] + 1) // 2)
    else:
        state["frames"] = None
    sigmas = transformer_options.get("sample_sigmas")
    if sigmas is not None:
        sigma = float(timestep.flatten()[0]) / 1000.0
        state["total_steps"] = max(1, int(sigmas.numel()) - 1)
        state["step"] = int((sigmas.float().cpu() - sigma).abs().argmin())
    return executor(x, timestep, context, transformer_options, minimax_payload=minimax_payload, **kwargs)


def patch_model(model, tau, log_diagnostics=True):
    state = {"tau": float(tau), "log_diagnostics": bool(log_diagnostics),
             "frames": None, "rows_per_frame": None,
             "step": 0, "total_steps": 1, "layers": 50, "calls_this_forward": 0,
             "guard_misses": 0, "warned_inactive": False,
             "fw_abs_tension": [], "fw_gamma": [], "fw_active": [], "fw_tst_time": []}
    patched = model.clone()
    patched.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
                                 "selflift_h3_tst", partial(_forward_wrapper, state))
    patched.model_options.setdefault("transformer_options", {})["optimized_attention_override"] = _attention_override(state)
    return patched
