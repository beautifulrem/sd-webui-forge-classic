"""Test support: a minimal initialised WebUI for the ``remi`` tests.

Default options (no config file), an empty style database and txt2img /
img2img script runners holding only the built-in processing scripts; no
model, extension or UI is loaded. Everything is restored on exit, so suites
that expect an uninitialised WebUI are unaffected.
"""

from __future__ import annotations

import importlib
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

PROCESSING_SCRIPTS = ("seed", "sampler", "refiner", "comments", "mahiro", "rescale_cfg")


def processing_script_classes(names=PROCESSING_SCRIPTS) -> list:
    from modules import scripts

    classes = []
    for name in names:
        module = importlib.import_module(f"modules.processing_scripts.{name}")
        for value in vars(module).values():
            if isinstance(value, type) and issubclass(value, scripts.Script) and value.__module__ == module.__name__:
                classes.append(scripts.ScriptClassData(value, module.__file__, str(Path(module.__file__).parents[2]), module))
    return classes


@contextmanager
def webui_state(options: dict | None = None, *, script_classes=None, postprocessing=()):
    """Yield ``SimpleNamespace(opts, txt2img, img2img, postproc)`` with
    ``options`` overriding the defaults. ``script_classes``: extra
    ``ScriptClassData`` (or classes) to load after the processing scripts;
    ``postprocessing``: the Extras operation classes (none by default)."""

    from modules import options as options_module
    from modules import shared, shared_options

    saved = {name: getattr(shared, name) for name in ("opts", "options_templates", "prompt_styles")}
    shared.options_templates = shared_options.options_templates
    shared.opts = options_module.Options(shared_options.options_templates, shared_options.restricted_opts)
    try:
        for key, value in (options or {}).items():
            shared.opts.data[key] = value

        import modules.processing  # noqa: F401 - the WebUI's import order (sd_models <-> processing)
        from modules import scripts, scripts_postprocessing, styles

        shared.prompt_styles = styles.StyleDatabase([str(Path(shared.data_path) / "__remi_tests_no_styles__.csv")])
        saved_scripts = (scripts.scripts_txt2img, scripts.scripts_img2img, scripts.scripts_postproc, list(scripts.scripts_data))
        try:
            scripts.scripts_data[:] = processing_script_classes()
            for extra in script_classes or ():
                if isinstance(extra, type):
                    extra = scripts.ScriptClassData(extra, f"{extra.__name__}.py", "", None)
                scripts.scripts_data.append(extra)
            scripts.scripts_txt2img = scripts.ScriptRunner()
            scripts.scripts_img2img = scripts.ScriptRunner()
            scripts.scripts_postproc = scripts_postprocessing.ScriptPostprocessingRunner()
            scripts.scripts_postproc.scripts = []
            for cls in postprocessing:
                operation = cls()
                operation.filename = f"{cls.__name__}.py"
                scripts.scripts_postproc.scripts.append(operation)
            scripts.scripts_txt2img.initialize_scripts(is_img2img=False)
            scripts.scripts_img2img.initialize_scripts(is_img2img=True)
            yield SimpleNamespace(opts=shared.opts, txt2img=scripts.scripts_txt2img, img2img=scripts.scripts_img2img, postproc=scripts.scripts_postproc)
        finally:
            scripts.scripts_txt2img, scripts.scripts_img2img, scripts.scripts_postproc, data = saved_scripts
            scripts.scripts_data[:] = data
    finally:
        for name, value in saved.items():
            setattr(shared, name, value)
