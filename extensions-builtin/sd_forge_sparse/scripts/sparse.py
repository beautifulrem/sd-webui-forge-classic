# https://github.com/Comfy-Org/ComfyUI/blob/v0.35.0/comfy_extras/nodes_sparse_attention.py

import logging
import re
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from backend.patcher.base import ModelPatcher

import torch

from backend.logging import setup_logger
from backend.quant_ops import ck
from modules import scripts
from remi import ui

logger = logging.getLogger("SolAttn")
setup_logger(logger)

HEAD_DIM = 128


class SparseAttnPatch:

    def __init__(self, tau, sigma_start, sigma_end, min_tokens, extra_tokens, dense_blocks, verbose, fused=True):
        self.fused = fused  # the Sol-Attn kernels run the sparse calls (else reference_sol_attn)
        self.tau = tau
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end
        self.min_tokens = min_tokens
        self.extra_tokens = extra_tokens
        self.dense_blocks = dense_blocks
        self.verbose = verbose

        self._logged: set[str] = set()

    def log_once(self, key: str, message: str, *, always: bool = False):
        """Log ``message`` once per ``key``: with "Verbose", or ``always`` (a warning)."""
        if (self.verbose or always) and key not in self._logged:
            self._logged.add(key)
            (logger.warning if always else logger.info)(message)

    def dense_reason(self, transformer_options: dict, tokens: int, block_index: int) -> str | None:
        if (sigmas := transformer_options.get("sigmas")) is not None:
            sigma = float(sigmas[0])
            if sigma > self.sigma_start or sigma < self.sigma_end:
                return 'sigma outside the "Timestep Range"'
        if tokens < self.min_tokens:
            return f"tokens ({tokens}) < min_tokens ({self.min_tokens})"
        if self.dense_blocks:
            if block_index is None:
                self.log_once("no_block_index", 'this model does not include block indices; "Dense Blocks" is ignored')
            elif block_index in self.dense_blocks:
                return f"block {block_index} is in dense_blocks"
        return None

    @staticmethod
    def ineligible(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, dim_head: int, fused: bool = True) -> str | None:
        """Why this call stays dense (None: it runs sparse). ``fused``: the
        Sol-Attn kernels must take it; otherwise the reference does, under
        the same rules for the tensors, so both run on the same calls."""
        if fused:
            if q.device.type != "cuda":
                return "CUDA device is required"
            if not ck.sol_attn_is_available(q.device):
                return "sol_attn is not available for this GPU"
        if q.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            return f'dtype "{q.dtype}" is not supported'
        if dim_head != HEAD_DIM:
            return f"head_dim ({dim_head}) != {HEAD_DIM}"
        if q.shape != k.shape or q.shape != v.shape:
            return "cross-attention"
        if k.dtype != q.dtype or v.dtype != q.dtype:
            return f"mixed dtypes: {q.dtype}/{k.dtype}/{v.dtype}"
        return None


def unavailable_reason(device) -> str | None:
    """Why the fused Sol-Attn kernels cannot run on ``device`` (None: they
    can). Checked once per sampling pass: elsewhere the run uses
    :func:`reference_sol_attn` and says so."""

    device = torch.device(device)
    if device.type != "cuda":
        return "CUDA device is required"
    try:
        available = ck.sol_attn_is_available(device)
    except Exception:
        available = False
    if not available:
        return "sol_attn is not available for this GPU"
    return None


REFERENCE_SCORE_BYTES = 512 * 2**20
"""The reference materialises its attention scores (fp32, tokens x tokens per
head): it runs as many heads at a time as fit in this (at least one)."""


def reference_sol_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, tau: float, scale: float | None = None) -> torch.Tensor:
    """Sol-Attn without the fused kernels: Comfy Kitchen's reference
    implementation (eager PyTorch, any device) of the same algorithm - the
    same block routing, in full precision instead of the kernels' INT8, so
    the result matches theirs up to rounding (and Extra Tokens, a detail of
    the kernels' routing, has no effect). ``(B, T, H, D)`` tensors.

    Every (batch, head) is independent, so it runs a few heads at a time to
    bound the memory of its dense score tensors; slower than regular
    attention."""

    implementation = ck.registry.get_implementation("sol_attn", backend="eager")
    batch, tokens, heads, _ = q.shape
    step = max(1, min(heads, REFERENCE_SCORE_BYTES // (tokens * tokens * 4)))
    out = torch.empty(v.shape, dtype=v.dtype, device=v.device)
    for b in range(batch):
        for h in range(0, heads, step):
            part = (slice(b, b + 1), slice(None), slice(h, h + step))
            out[part] = implementation(q[part], k[part], v[part], tau=tau, scale=scale)
    return out


INFOTEXT_PREFIX = "Sparse attention"

# What unavailable_reason() decides, said before a run instead of after it
REQUIREMENT_NOTE = (
    "**Faster on supported GPUs only:** the fused Comfy Kitchen Sol-Attn kernels need an NVIDIA GPU of compute capability "
    "8.0 or newer (RTX 30 series and later) or an AMD GPU with matrix cores (ROCm). Elsewhere (Apple Silicon (MPS), the "
    "CPU, older GPUs) the run uses Comfy Kitchen's reference implementation of the same algorithm: the same sparse-attention "
    "image (up to the kernels' rounding), but slower than regular attention, and the job says so."
)


def make_attention_override(patch: "SparseAttnPatch", prev_attn: Callable) -> Callable:

    def override(func: Callable, q, k, v, heads, **kwargs):

        def dense():
            args = (q, k, v, heads)
            return func(*args, **kwargs) if prev_attn is None else prev_attn(func, *args, **kwargs)

        if kwargs.get("mask", None) is not None:
            return dense()

        transformer_options: dict = kwargs.get("transformer_options") or {}
        tokens: int = q.shape[2 if kwargs.get("skip_reshape", False) else 1]

        reason = patch.dense_reason(transformer_options, tokens, transformer_options.get("block_index", None))
        if reason is not None:
            patch.log_once(("dense", tokens, reason), f"dense: {reason}")
            return dense()

        if kwargs.get("skip_reshape", False):
            b, _, _, dim_head = q.shape
            qs, ks, vs = (t.transpose(1, 2) for t in (q, k, v))
        else:
            b, _, dim_head = q.shape
            dim_head //= heads
            qs, ks, vs = (t.view(b, -1, heads, dim_head) for t in (q, k, v))

        reason = patch.ineligible(qs, ks, vs, dim_head, patch.fused)
        if reason is not None:
            patch.log_once(("ineligible", tuple(qs.shape), reason), f"dense: {reason}")
            return dense()

        if patch.fused:
            if q.dtype == torch.float32:
                qs, ks, vs = (t.to(dtype=torch.bfloat16) for t in (qs, ks, vs))
            out = ck.sol_attn(qs, ks, vs, tau=patch.tau, scale=kwargs.get("scale"), token_aug=patch.extra_tokens).to(q.dtype)
        else:
            try:
                out = reference_sol_attn(qs, ks, vs, tau=patch.tau, scale=kwargs.get("scale"))
            except RuntimeError as error:  # out of memory, or too many tokens for the reference
                patch.log_once(("reference failed", tuple(qs.shape)), f"dense: the reference sparse attention failed for {tuple(qs.shape)}: {error}", always=True)
                return dense()
        patch.log_once(("sparse", tuple(qs.shape)), "sparse" if patch.fused else "sparse (reference)")

        if kwargs.get("skip_output_reshape", False):
            return out.transpose(1, 2)
        else:
            return out.reshape(b, -1, heads * dim_head)

    return override


class SparseAttentionForForge(scripts.Script):

    def title(self):
        return "Sparse Attention Integrated"

    def show(self, *args, **kwargs):
        return scripts.AlwaysVisible

    def params(self, is_img2img):
        return ui.Section(
            self.title(),
            # Like Refiner: an image without the record ran without sparse attention
            enable=ui.Toggle("enable", False, label=self.title(), paste=lambda d: f"{INFOTEXT_PREFIX} tau" in d),
            children=[
                ui.Markdown(REQUIREMENT_NOTE),
                ui.Row(
                    ui.Slider("tau", 0.0, 4.0, 1.25, 0.05, label="Tau", info="higher = faster & worse quality", infotext=f"{INFOTEXT_PREFIX} tau"),
                    ui.RangeSlider("percent", 0.0, 1.0, (0.15, 0.85), 0.05, label="Timestep Range", infotext=f"{INFOTEXT_PREFIX} range"),
                ),
                ui.Row(
                    ui.Slider("min_tokens", 0, 32768, 4096, 512, label="Tokens Threshold", info="model & resolution dependent ; higher = slower & better quality", infotext=f"{INFOTEXT_PREFIX} min tokens"),
                    ui.Slider("extra_tokens", 0, 256, 0, 64, label="Extra Tokens", info="higher = slower & better quality", infotext=f"{INFOTEXT_PREFIX} extra tokens"),
                ),
                ui.Text("dense_blocks", "", max_lines=1, placeholder="0, 1, 25-27", label="Dense Blocks", info="blocks to keep dense (i.e. slower / better quality)", paste=lambda d: d.get(f"{INFOTEXT_PREFIX} dense blocks", "" if f"{INFOTEXT_PREFIX} tau" in d else None)),
                ui.Toggle("verbose", False, label="Verbose (Debug)"),
            ],
        )

    def process_before_every_sampling(self, p, enable: bool, tau: float = 1.25, percent: tuple[float, float] = (0.15, 0.85), min_tokens: int = 4096, extra_tokens: int = 0, dense_blocks: str = "", verbose: bool = False, **kwargs):
        if not enable:
            return

        from modules import devices

        if (reason := unavailable_reason(devices.device)) is not None:
            # The reference runs instead: said in the job's warnings (p.comment),
            # as it is slower than regular attention
            message = f"{self.title()} used the reference implementation ({reason}): slower than regular attention"
            if callable(comment := getattr(p, "comment", None)):
                comment(message)
            logger.warning(message)

        unet = p.sd_model.forge_objects.unet

        patched_unet = self._apply_block_sparse_attention(
            model=unet,
            tau=tau,
            start_percent=percent[0],
            end_percent=percent[1],
            min_tokens=min_tokens,
            extra_tokens=extra_tokens,
            dense_blocks=self._parse_block_list(dense_blocks),
            verbose=verbose,
            fused=reason is None,
        )

        p.sd_model.forge_objects.unet = patched_unet

        extra = getattr(p, "extra_generation_params", None)
        if extra is not None:
            extra[f"{INFOTEXT_PREFIX} tau"] = tau
            extra[f"{INFOTEXT_PREFIX} range"] = f"{percent[0]}, {percent[1]}"
            extra[f"{INFOTEXT_PREFIX} min tokens"] = min_tokens
            extra[f"{INFOTEXT_PREFIX} extra tokens"] = extra_tokens
            if dense_blocks.strip():
                extra[f"{INFOTEXT_PREFIX} dense blocks"] = dense_blocks.strip()

    @staticmethod
    def _apply_block_sparse_attention(model: "ModelPatcher", tau: float, start_percent: float, end_percent: float, min_tokens: int, extra_tokens: int, dense_blocks: set[int], verbose: bool, fused: bool = True) -> "ModelPatcher":
        model_sampling = model.model.predictor

        patch = SparseAttnPatch(
            tau=tau,
            sigma_start=float(model_sampling.percent_to_sigma(start_percent)),
            sigma_end=float(model_sampling.percent_to_sigma(end_percent)),
            min_tokens=min_tokens,
            extra_tokens=extra_tokens,
            dense_blocks=dense_blocks,
            verbose=verbose,
            fused=fused,
        )

        m = model.clone()

        prev = m.model_options["transformer_options"].get("optimized_attention_override", None)
        override = make_attention_override(patch, prev)
        m.model_options["transformer_options"]["optimized_attention_override"] = override

        return m

    @staticmethod
    def _parse_block_list(text: str) -> set[int]:
        blocks = set()

        for part in re.findall(r"\d+\s*-\s*\d+|\d+", text or ""):
            if "-" in part:
                a, b = (int(x) for x in part.split("-"))
                blocks.update(range(min(a, b), max(a, b) + 1))
            else:
                blocks.add(int(part))

        return blocks
