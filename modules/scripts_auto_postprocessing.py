"""Extras operations run in txt2img / img2img (setting
``postprocessing_enable_in_main_ui``).

Each selected ``ScriptPostprocessing`` is wrapped in an always-on
``scripts.Script`` whose ``params()`` is the operation's own layout and whose
actions are the operation's actions, so the generation form renders and drives
it like in Extras. After each image is composited the wrapper runs the
operation's ``process_firstpass`` and ``process`` with the values by name, and
the operation's infotext joins the generation parameters.
"""

from modules import scripts, scripts_postprocessing, shared
from remi.ui.actions import ActionSpec, actions_of


class ScriptPostprocessingForMainUI(scripts.Script):
    def __init__(self, script_postproc):
        self.script: scripts_postprocessing.ScriptPostprocessing = script_postproc

    def title(self):
        return self.script.name

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def params(self, is_img2img):
        from remi.core.extras import legacy_ui_only

        if legacy_ui_only(self.script):
            raise TypeError(f"the Extras operation {self.script.name!r} declares ui() instead of params()")
        return self.script.params()

    def postprocess_image_after_composite(self, p, script_pp, *args):
        layout = getattr(self, "layout", None)
        args_dict = dict(zip(layout.names if layout is not None else (), args))

        pp = scripts_postprocessing.PostprocessedImage(script_pp.image)
        pp.info = {}
        self.script.process_firstpass(pp, **args_dict)
        self.script.process(pp, **args_dict)
        p.extra_generation_params.update(pp.info)
        script_pp.image = pp.image


def _forward(spec: ActionSpec):
    def forward(self, ctx, values):
        return spec.func.__get__(self.script, type(self.script))(ctx, values)

    forward.__name__ = spec.name
    forward.__remi_action__ = ActionSpec(spec.name, forward, spec.gpu)
    return forward


def wrapper_class(script_class) -> type:
    """A ``ScriptPostprocessingForMainUI`` subclass that also carries the
    operation's ``@ui.action`` methods (forwarded to the wrapped operation),
    so its buttons and ``on_change`` actions work in the generation form."""

    members = {f"_action_{name}": _forward(spec) for name, spec in actions_of(script_class).items()}
    return type(f"MainUI{script_class.__name__}", (ScriptPostprocessingForMainUI,), members)


def create_auto_preprocessing_script_data():
    res = []

    for name in shared.opts.postprocessing_enable_in_main_ui:
        script = next(iter([x for x in scripts.postprocessing_scripts_data if x.script_class.name == name]), None)
        if script is None:
            continue

        cls = wrapper_class(script.script_class)
        constructor = lambda s=script, cls=cls: cls(s.script_class())
        res.append(scripts.ScriptClassData(script_class=constructor, path=script.path, basedir=script.basedir, module=script.module))

    return res
