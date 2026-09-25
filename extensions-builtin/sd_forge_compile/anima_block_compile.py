"""Transactional per-block torch.compile support for Anima models."""

from __future__ import annotations

from functools import wraps

import torch


_BLOCK_CONFIG_KEY = "_forge_anima_block_compile_config"
_COMPILED_BLOCKS_KEY = "_forge_anima_compiled_blocks"
_ORIGINAL_APPLY_KEY = "_forge_anima_block_original_apply_model"
_WRAPPER_MARKER = "_forge_anima_block_compile_wrapper"
_FAILED_KEY = "_forge_anima_block_compile_failed"
_FALLBACK_HANDLER_KEY = "_forge_anima_block_compile_fallback_handler"
_FALLBACK_REASON_KEY = "_forge_anima_block_compile_fallback_reason"


# Per-generation transformer_options entries the blocks act on. The compile
# config drops guards on transformer_options (they change every step), so a
# graph traced without these would silently keep ignoring them: calls that
# carry any run the eager blocks instead.
_EAGER_OPTION_KEYS = ("anima_attention_modifiers", "negpip_mask", "negpip_context_mask")


def patched_inside(module_ids) -> bool:
    """True while a forward override (Regional, FreeFuse, Artist Mixer,
    NegPiP...) is installed on any module whose id is in ``module_ids``:
    compiled graphs traced without it would ignore or keep recompiling it."""
    from modules.forward_override import overridden_modules

    return any(id(module) in module_ids for module in overridden_modules())


def needs_eager(args, kwargs, module_ids) -> bool:
    return _needs_eager_blocks(args, kwargs) or patched_inside(module_ids)


def _needs_eager_blocks(args, kwargs) -> bool:
    # NegPiP's mask arrives as a conditioning kwarg; its DiT hook moves it
    # into transformer_options only after the compiled blocks are in place.
    if kwargs.get("c_negpip_mask") is not None:
        return True
    options = kwargs.get("transformer_options")
    if options is None and len(args) > 5:
        options = args[5]
    return isinstance(options, dict) and any(options.get(key) is not None and options.get(key) != () for key in _EAGER_OPTION_KEYS)


def _is_compile_failure(error: Exception) -> bool:
    module = type(error).__module__
    name = type(error).__name__
    return module.startswith(("torch._dynamo", "torch._inductor")) or name in {
        "BackendCompilerFailed",
        "InductorError",
    }


class AnimaBlockCompileManager:
    """Compile Anima transformer blocks and swap them only during inference."""

    @staticmethod
    def install(kmodel, compile_config: dict, config_name: str, on_fallback=None) -> None:
        original_apply_model = kmodel.apply_model
        setattr(kmodel, _ORIGINAL_APPLY_KEY, original_apply_model)
        setattr(kmodel, _BLOCK_CONFIG_KEY, config_name)
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
            block_ids = {id(module) for block in blocks for module in block.modules()}
            if needs_eager(args, kwargs, block_ids):
                return original_apply_model(*args, **kwargs)

            compiled = getattr(kmodel, _COMPILED_BLOCKS_KEY, None)
            if compiled is None or len(compiled) != len(blocks):
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
                # it can make its eager fallback explicit and persistent.
                with torch._dynamo.config.patch(suppress_errors=False):
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
            _BLOCK_CONFIG_KEY,
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
