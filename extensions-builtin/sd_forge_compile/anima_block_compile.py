"""Transactional per-block torch.compile support for Anima models."""

from __future__ import annotations

from functools import wraps

import torch

from backend.nn.anima_attention import ANIMA_ATTENTION_MODIFIERS, active_attention_modifiers
from modules.anima_support import NEGPIP_CONTEXT_MASK_KEY, NEGPIP_MASK_KEY

_COMPILED_BLOCKS_KEY = "_forge_anima_compiled_blocks"
_ORIGINAL_APPLY_KEY = "_forge_anima_block_original_apply_model"
_WRAPPER_MARKER = "_forge_anima_block_compile_wrapper"
_FAILED_KEY = "_forge_anima_block_compile_failed"
_FALLBACK_HANDLER_KEY = "_forge_anima_block_compile_fallback_handler"
_FALLBACK_REASON_KEY = "_forge_anima_block_compile_fallback_reason"


# Per-generation transformer_options entries the blocks act on. The compile
# config drops guards on transformer_options (they change every step), so a
# graph traced without these would silently keep ignoring them: calls that
# carry any run the eager blocks instead. That includes an attention override
# (Sparse Attention): it reads the per-block ``block_index``, which a graph
# would bake in from the block it was traced on.
_EAGER_OPTION_KEYS = (ANIMA_ATTENTION_MODIFIERS, NEGPIP_MASK_KEY, NEGPIP_CONTEXT_MASK_KEY, "optimized_attention_override")


def patched_inside(root) -> bool:
    """True while a forward override (Regional, FreeFuse, Artist Mixer,
    NegPiP...) is installed on ``root`` or any module inside it: compiled
    graphs traced without it would ignore or keep recompiling it."""
    from modules.forward_override import overridden_modules

    overridden = {id(module) for module in overridden_modules()}
    # Usually nothing is overridden: skip walking the (~850) submodules.
    return bool(overridden) and any(id(module) in overridden for module in root.modules())


def needs_eager(args, kwargs, root) -> bool:
    return _needs_eager_blocks(args, kwargs) or patched_inside(root)


def _transformer_options(args, kwargs):
    options = kwargs.get("transformer_options")
    if options is None and len(args) > 5:
        options = args[5]
    return options


def without_inactive_modifiers(args, kwargs):
    """Drop attention modifiers that do nothing on this call (NAG outside
    its sigma window), so the call can still run the compiled blocks.
    Returns ``(args, kwargs)``, with a copied options dict if any dropped."""

    options = _transformer_options(args, kwargs)
    if not isinstance(options, dict) or not options.get(ANIMA_ATTENTION_MODIFIERS):
        return args, kwargs
    active = active_attention_modifiers(options)
    if len(active) == len(options[ANIMA_ATTENTION_MODIFIERS]):
        return args, kwargs
    options = {**options, ANIMA_ATTENTION_MODIFIERS: active}
    if kwargs.get("transformer_options") is not None:
        return args, {**kwargs, "transformer_options": options}
    return (*args[:5], options, *args[6:]), kwargs


def _needs_eager_blocks(args, kwargs) -> bool:
    # NegPiP's mask arrives as a conditioning kwarg; its DiT hook moves it
    # into transformer_options only after the compiled blocks are in place.
    if kwargs.get("c_negpip_mask") is not None:
        return True
    options = _transformer_options(args, kwargs)
    return isinstance(options, dict) and any(options.get(key) is not None and options.get(key) != () for key in _EAGER_OPTION_KEYS)


def _is_compile_failure(error: Exception) -> bool:
    module = type(error).__module__
    name = type(error).__name__
    return module.startswith(("torch._dynamo", "torch._inductor")) or name in {
        "BackendCompilerFailed",
        "InductorError",
    }


def block_compile_config(guard_filter_fn) -> dict:
    """torch.compile options for the per-block preset. Every block shares
    Block.forward's code object, so with ``dynamic=False`` each new
    (batch, H, W) adds a cache entry until Dynamo's recompile limit, then the
    blocks run eager. ``dynamic=None`` marks a changed size dynamic on its
    first recompile, so later sizes reuse that graph, except where Dynamo
    specializes: a size of 1 always gets a graph of its own (batch 1 at CFG 1
    or outside the CFG range; the single frame of a pass without reference
    images, against T=2 with one)."""
    return dict(backend="inductor", dynamic=None, fullgraph=False, options={"guard_filter_fn": guard_filter_fn})


# Graph variants the shared Block.forward code object may collect before the
# per-block preset falls back to eager for good. Dynamo's default (8) counts
# every variant of all blocks together: the static first graph and its
# dynamic successor, each again for batch 1 and for T=2 reference frames
# and per grad mode, fill it during ordinary use. The size sequence in
# tests/test_anima_block_compile_dynamo.py reaches 11 variants under
# inference_mode and 15 once no_grad calls join. 32 leaves about twice that
# while still stopping a guard that recompiles on every step well before
# Dynamo's accumulated limit (256).
RECOMPILE_LIMIT = 32


def _recompile_limit_patch() -> tuple[object, dict]:
    """The config module and settings that make hitting RECOMPILE_LIMIT raise.

    The launcher installs PyTorch 2.13 (2.11 for Nunchaku), whose public
    ``torch.compiler.config`` carries both settings. Older installs only have
    them in ``torch._dynamo.config``, before 2.7 under their cache-size names."""
    public = getattr(torch.compiler, "config", None)
    if all(hasattr(public, name) for name in ("recompile_limit", "fail_on_recompile_limit_hit")):
        return public, dict(recompile_limit=RECOMPILE_LIMIT, fail_on_recompile_limit_hit=True)
    dynamo = torch._dynamo.config
    if hasattr(dynamo, "recompile_limit"):
        return dynamo, dict(recompile_limit=RECOMPILE_LIMIT, fail_on_recompile_limit_hit=True)
    return dynamo, dict(cache_size_limit=RECOMPILE_LIMIT, fail_on_cache_limit_hit=True)


class AnimaBlockCompileManager:
    """Compile Anima transformer blocks and swap them only during inference."""

    @staticmethod
    def install(kmodel, compile_config: dict, on_fallback=None) -> None:
        original_apply_model = kmodel.apply_model
        setattr(kmodel, _ORIGINAL_APPLY_KEY, original_apply_model)
        setattr(kmodel, _FALLBACK_HANDLER_KEY, on_fallback)

        def fallback(error: Exception) -> None:
            setattr(kmodel, _FAILED_KEY, True)
            setattr(kmodel, _FALLBACK_REASON_KEY, type(error).__name__)
            handler = getattr(kmodel, _FALLBACK_HANDLER_KEY, None)
            if handler is not None:
                handler(error)

        @wraps(original_apply_model)
        def apply_model_with_compiled_blocks(*args, **kwargs):
            if getattr(kmodel, _FAILED_KEY, False):
                return original_apply_model(*args, **kwargs)
            diffusion_model = kmodel.diffusion_model
            blocks = getattr(diffusion_model, "blocks", None)
            if not isinstance(blocks, torch.nn.ModuleList) or not blocks:
                return original_apply_model(*args, **kwargs)
            args, kwargs = without_inactive_modifiers(args, kwargs)
            if needs_eager(args, kwargs, blocks):
                return original_apply_model(*args, **kwargs)

            compiled = getattr(kmodel, _COMPILED_BLOCKS_KEY, None)
            if compiled is None or len(compiled) != len(blocks):
                # torch.compile is lazy: real compiler failures surface at the
                # first call, below. Defensive only: at construction it raises
                # a torch._dynamo error just for an unknown backend name
                # (lookup_backend's InvalidBackend); the preset uses inductor.
                try:
                    compiled = tuple(torch.compile(block, **compile_config) for block in blocks)
                    setattr(kmodel, _COMPILED_BLOCKS_KEY, compiled)
                except Exception as error:
                    if not _is_compile_failure(error):
                        raise
                    fallback(error)
                    return original_apply_model(*args, **kwargs)

            originals = tuple(blocks)
            result = None
            compile_error = None
            try:
                for index, block in enumerate(compiled):
                    blocks[index] = block
                # The integrated whole-model preset keeps Dynamo's forgiving
                # default. Per-block mode needs the lazy compiler exception so
                # it can make its eager fallback explicit and persistent;
                # that includes the recompile limit, which otherwise runs
                # the blocks eager without a word, raised to RECOMPILE_LIMIT
                # for this preset only.
                limit_config, limit_settings = _recompile_limit_patch()
                with torch._dynamo.config.patch(suppress_errors=False), limit_config.patch(**limit_settings):
                    result = original_apply_model(*args, **kwargs)
            except Exception as error:
                if not _is_compile_failure(error):
                    raise
                compile_error = error
            finally:
                for index, block in enumerate(originals):
                    blocks[index] = block
            if compile_error is not None:
                if hasattr(kmodel, _COMPILED_BLOCKS_KEY):
                    delattr(kmodel, _COMPILED_BLOCKS_KEY)
                fallback(compile_error)
                return original_apply_model(*args, **kwargs)
            return result

        setattr(apply_model_with_compiled_blocks, _WRAPPER_MARKER, True)
        kmodel.apply_model = apply_model_with_compiled_blocks

    @staticmethod
    def remove(kmodel) -> None:
        original = getattr(kmodel, _ORIGINAL_APPLY_KEY, None)
        if original is not None and getattr(kmodel.apply_model, _WRAPPER_MARKER, False):
            kmodel.apply_model = original
        for name in (
            _COMPILED_BLOCKS_KEY,
            _ORIGINAL_APPLY_KEY,
            _FAILED_KEY,
            _FALLBACK_HANDLER_KEY,
            _FALLBACK_REASON_KEY,
        ):
            if hasattr(kmodel, name):
                delattr(kmodel, name)

    @staticmethod
    def is_installed(kmodel) -> bool:
        return bool(getattr(kmodel.apply_model, _WRAPPER_MARKER, False))

    @staticmethod
    def update_fallback_handler(kmodel, handler) -> None:
        setattr(kmodel, _FALLBACK_HANDLER_KEY, handler)

    @staticmethod
    def fallback_reason(kmodel) -> str | None:
        return getattr(kmodel, _FALLBACK_REASON_KEY, None)
