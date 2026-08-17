# https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_torch_compile.py

import logging
import shutil
import sys
from functools import wraps
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.modules.k_model import KModel

import gradio as gr
import torch

from backend.args import args as cmd_args
from backend.logging import setup_logger
from backend.utils import get_attr, set_attr_raw
from modules import scripts
from modules.anima_support import is_anima_engine
from modules.anima_presets import register_preset_control
from anima_block_compile import AnimaBlockCompileManager

try:
    import triton  # noqa: F401 -- importing is the backend availability probe
except ImportError:
    TRITON_AVAILABLE = False
else:
    TRITON_AVAILABLE = True

_COMPILE_CONFIG_KEY = "_torch_compile_config"
_COMPILE_WRAPPER_KEY = "_torch_compile_wrapper"
_ORIG_APPLY_KEY = "_orig_apply_model"

logger = logging.getLogger("compile")
setup_logger(logger)


def skip_torch_compile_dict(guard_entries):
    return [("transformer_options" not in entry.name) for entry in guard_entries]


def _windows_cpp_compiler_available() -> bool:
    if sys.platform != "win32":
        return True
    return any(shutil.which(name) for name in ("cl", "clang-cl", "g++"))


class TorchCompileForForge(scripts.Script):
    sorting_priority = 99999

    def __init__(self):
        torch._dynamo.config.cache_size_limit = 256
        torch._dynamo.config.suppress_errors = True

    def title(self):
        return "Torch Compile Integrated"

    def show(self, is_img2img):
        return scripts.AlwaysVisible if TRITON_AVAILABLE else None

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            preset = gr.Dropdown(
                label="Preset",
                value="Automatic",
                choices=[
                    "Automatic",
                    "Disable",
                    "Anima per-block",
                    "guard_filter_fn",
                    "dynamic",
                    "max-autotune",
                    "max-autotune-no-cudagraphs",
                    "reduce-overhead",
                ],
                info='"Automatic" maintains the current compile status',
            )

            _dynamic = "Support any Resolution / Batch Size"
            _indynamic = "Require recompilation if Resolution / Batch Size is changed"
            _no_malloc = "Does not work with --cuda-malloc"

            gr.Markdown(rf"""
**torch.compile** speeds up the inference by compiling the model ahead of time
- **guard_filter_fn:** Compile the Fastest ; {_indynamic}
- **dynamic:** {_dynamic} ; Slower to Compile
- **max-autotune:** Best Runtime Speed ; {_indynamic} ; {_no_malloc}
- **max-autotune-no-cudagraphs:** {_dynamic} ; Faster than **dynamic** ; Even Slower to Compile
- **reduce-overhead:** Similar to **max-autotune** ; {_indynamic} ; {_no_malloc}
- **Anima per-block:** Compile each Anima transformer block separately; fixed resolution/batch, lower graph-break risk than whole-model compile
            """)

        register_preset_control(self.tabname, "compile.preset", preset)

        return [preset]

    def process_batch(self, p, preset: str, **kwargs):
        kmodel: "KModel" = p.sd_model.forge_objects.unet.model
        prev_config: tuple[str] = getattr(kmodel, _COMPILE_CONFIG_KEY, None)

        def compile_fallback(error):
            message = f"Anima per-block fell back to eager: {type(error).__name__}"
            p.extra_generation_params["Torch compile"] = message
            logger.error(
                message,
                exc_info=(type(error), error, error.__traceback__),
            )

        if preset == "Automatic":
            if AnimaBlockCompileManager.is_installed(kmodel):
                AnimaBlockCompileManager.update_fallback_handler(
                    kmodel, compile_fallback
                )
                reason = AnimaBlockCompileManager.fallback_reason(kmodel)
                if reason is not None:
                    p.extra_generation_params["Torch compile"] = (
                        f"Anima per-block using eager fallback: {reason}"
                    )
            return

        if preset == "Disable":
            self._remove_compile_wrapper(kmodel)
            return

        if preset == "Anima per-block" and not is_anima_engine(p.sd_model):
            logger.warning("Anima per-block compile was ignored because the active model is not Anima")
            p.extra_generation_params["Torch compile"] = "Anima per-block ignored: non-Anima model"
            return

        if preset == "Anima per-block" and not _windows_cpp_compiler_available():
            self._remove_compile_wrapper(kmodel)
            message = "Anima per-block disabled: no C++ compiler found (install Visual Studio Build Tools)"
            logger.error(message)
            p.extra_generation_params["Torch compile"] = message
            return

        if preset in ("max-autotune", "reduce-overhead") and cmd_args.cuda_malloc:
            logger.error(f"{preset} does not support --cuda-malloc\nModel is not compiled...")
            return

        if prev_config == preset and (
            getattr(kmodel, _ORIG_APPLY_KEY, None) is not None
            or AnimaBlockCompileManager.is_installed(kmodel)
        ):
            if preset == "Anima per-block":
                AnimaBlockCompileManager.update_fallback_handler(
                    kmodel, compile_fallback
                )
                reason = AnimaBlockCompileManager.fallback_reason(kmodel)
                if reason is not None:
                    p.extra_generation_params["Torch compile"] = (
                        f"Anima per-block using eager fallback: {reason}"
                    )
            return

        if prev_config is not None:
            self._remove_compile_wrapper(kmodel)

        match preset:
            case "Anima per-block":
                config = dict(
                    backend="inductor",
                    dynamic=False,
                    fullgraph=False,
                    options={"guard_filter_fn": skip_torch_compile_dict},
                )
            case "guard_filter_fn":
                config = dict(backend="inductor", dynamic=False, fullgraph=False, options={"guard_filter_fn": skip_torch_compile_dict})
            case "dynamic":
                config = dict(backend="inductor", dynamic=True, fullgraph=False, options={"guard_filter_fn": skip_torch_compile_dict})
            case "max-autotune":
                config = dict(backend="inductor", dynamic=False, fullgraph=False, options={"guard_filter_fn": skip_torch_compile_dict, "coordinate_descent_tuning": True, "max_autotune": True, "triton.cudagraphs": True})
            case "max-autotune-no-cudagraphs":
                config = dict(backend="inductor", dynamic=True, fullgraph=False, options={"guard_filter_fn": skip_torch_compile_dict, "coordinate_descent_tuning": True, "max_autotune": True})
            case "reduce-overhead":
                config = dict(backend="inductor", mode="reduce-overhead", dynamic=False, fullgraph=False, options={"guard_filter_fn": skip_torch_compile_dict})

        if preset == "Anima per-block":
            AnimaBlockCompileManager.install(
                kmodel,
                config,
                preset,
                on_fallback=compile_fallback,
            )
        else:
            self._wrap_apply_model(kmodel, config)
        setattr(kmodel, _COMPILE_CONFIG_KEY, preset)

        logger.info(f"Model Compiled ({preset})")

    @staticmethod
    def _wrap_apply_model(kmodel: "KModel", compile_config: dict):
        original_apply_model = kmodel.apply_model
        setattr(kmodel, _ORIG_APPLY_KEY, original_apply_model)

        @wraps(original_apply_model)
        def apply_model_with_compile(*args, **kwargs):
            orig_model = get_attr(kmodel, "diffusion_model")

            if not hasattr(kmodel, "_forge_compiled_model"):
                setattr(kmodel, "_forge_compiled_model", torch.compile(orig_model, **compile_config))

            compiled = getattr(kmodel, "_forge_compiled_model")
            set_attr_raw(kmodel, "diffusion_model", compiled)

            try:
                return original_apply_model(*args, **kwargs)
            finally:
                set_attr_raw(kmodel, "diffusion_model", orig_model)

        kmodel.apply_model = apply_model_with_compile
        setattr(kmodel.apply_model, _COMPILE_WRAPPER_KEY, True)

    @staticmethod
    def _remove_compile_wrapper(kmodel: "KModel"):
        AnimaBlockCompileManager.remove(kmodel)
        if hasattr(kmodel, "_forge_compiled_model"):
            delattr(kmodel, "_forge_compiled_model")

        if (orig := getattr(kmodel, _ORIG_APPLY_KEY, None)) is not None:
            if getattr(kmodel.apply_model, _COMPILE_WRAPPER_KEY, False):
                kmodel.apply_model = orig

        for attr in (_ORIG_APPLY_KEY, _COMPILE_CONFIG_KEY):
            if hasattr(kmodel, attr):
                delattr(kmodel, attr)
