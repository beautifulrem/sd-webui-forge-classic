import importlib
import logging
import os
import sys
import warnings

from modules.timer import startup_timer


def shush():
    logging.getLogger("diffusers.configuration_utils").setLevel(logging.ERROR)
    logging.getLogger("torch.distributed.nn").setLevel(logging.ERROR)
    logging.getLogger("transformers.dynamic_module_utils").setLevel(logging.ERROR)
    logging.getLogger("xformers").addFilter(lambda record: "triton" not in record.getMessage().lower())
    warnings.filterwarnings(action="ignore", category=UserWarning, module="torchvision.transforms.functional_tensor")
    warnings.filterwarnings(action="ignore", category=UserWarning, module="torchvision")
    startup_timer.record("filter logging")


def imports():
    # Extensions not ported yet may still import Gradio; keep it offline then
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

    from modules import errors, paths, timer  # noqa: F401

    startup_timer.record("setup paths")

    from modules import shared_init

    shared_init.initialize()
    startup_timer.record("shared init")

    from modules import processing  # noqa: F401

    startup_timer.record("misc. imports")


def check_versions():
    from modules.shared_cmd_options import cmd_opts

    if cmd_opts.skip_version_check:
        return

    import torch
    from packaging import version

    from modules import errors

    expected_torch = "2.10.0" if torch.cuda.is_available() and torch.cuda.get_device_properties().major < 8 else "2.13.0"
    if version.parse(torch.__version__) < version.parse(expected_torch):
        errors.print_error_explanation(
            f"You are running PyTorch {torch.__version__}, which is outdated.\n\n"
            "Delete the venv folder to do a reinstall is recommended.\n"
            "Use --skip-version-check commandline argument to disable the version checks."
        )

    startup_timer.record("version check")


def initialize():
    from modules import initialize_util

    initialize_util.fix_torch_version()
    initialize_util.fix_asyncio_event_loop_policy()
    initialize_util.validate_tls_options()
    initialize_util.configure_sigint_handler()
    initialize_util.configure_opts_onchange()

    from modules import sd_models

    sd_models.setup_model()

    from modules import codeformer_model
    from modules.shared_cmd_options import cmd_opts

    codeformer_model.setup_model(cmd_opts.codeformer_models_path)
    startup_timer.record("setup codeformer")

    from modules import gfpgan_model

    gfpgan_model.setup_model(cmd_opts.gfpgan_models_path)
    startup_timer.record("setup gfpgan")

    initialize_rest(reload_script_modules=False)


def initialize_rest(*, reload_script_modules=False):
    """
    Called both from initialize() and when reloading the webui.
    """
    from modules import sd_samplers
    from modules.shared_cmd_options import cmd_opts

    sd_samplers.set_samplers()
    startup_timer.record("set samplers")

    from modules import extensions

    extensions.list_extensions()
    startup_timer.record("list extensions")

    from modules import initialize_util

    initialize_util.restore_config_state_file()
    startup_timer.record("restore config state file")

    from modules import scripts, shared, upscaler

    if cmd_opts.ui_debug_mode:
        shared.sd_upscalers = upscaler.UpscalerLanczos().scalers
        scripts.load_scripts()
        return

    from modules import sd_models

    sd_models.list_models()
    startup_timer.record("list SD models")

    from modules import localization

    localization.list_localizations(cmd_opts.localizations_dir)
    startup_timer.record("list localizations")

    with startup_timer.subcategory("load scripts"):
        scripts.load_scripts()

    if reload_script_modules and shared.opts.enable_reloading_ui_scripts:
        for module in [module for name, module in sys.modules.items() if name.startswith("modules.ui")]:
            importlib.reload(module)
        startup_timer.record("reload script modules")

    from modules import modelloader

    modelloader.load_upscalers()
    startup_timer.record("load upscalers")

    from modules import sd_vae

    sd_vae.refresh_vae_list()
    startup_timer.record("refresh VAE")

    from modules import sd_unet

    sd_unet.list_unets()
    startup_timer.record("scripts list_unets")

    from modules import extra_networks

    extra_networks.initialize()
    startup_timer.record("initialize extra networks")
