import inspect
import os
import re
import sys
from collections import namedtuple
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modules import errors, extensions, paths, script_callbacks, script_loading, scripts_postprocessing, shared, timer, util

if TYPE_CHECKING:
    from remi.ui import Context, ScriptLayout
    from remi.ui.types import ScriptEntry

topological_sort = util.topological_sort

AlwaysVisible = object()


class ScriptAbort(RuntimeError):
    """Stop the current generation when continuing would produce invalid output.

    Only ``before_process_batch`` re-raises it (other hooks report and go on):
    that is where per-batch settings, e.g. conflicting LoRA rules, are known.
    """


class MaskBlendArgs:
    def __init__(self, current_latent, nmask, init_latent, mask, blended_latent, denoiser=None, sigma=None):
        self.current_latent = current_latent
        self.nmask = nmask
        self.init_latent = init_latent
        self.mask = mask
        self.blended_latent = blended_latent

        self.denoiser = denoiser
        self.is_final_blend = denoiser is None
        self.sigma = sigma


class PostSampleArgs:
    def __init__(self, samples):
        self.samples = samples


class PostprocessImageArgs:
    def __init__(self, image, index):
        self.image = image
        self.index = index


class PostProcessMaskOverlayArgs:
    def __init__(self, index, mask_for_overlay, overlay_image):
        self.index = index
        self.mask_for_overlay = mask_for_overlay
        self.overlay_image = overlay_image


class PostprocessBatchListArgs:
    def __init__(self, images):
        self.images = images


class Script:
    name = None
    """script's internal name derived from title"""

    section = None
    """name of UI section that the script's controls will be placed into"""

    filename = None
    args_from = None
    args_to = None
    alwayson = False

    is_txt2img = False
    is_img2img = False
    tabname = None

    api_info = None
    """Generated value of type modules.api.models.ScriptInfo with information about the script for API"""

    setup_for_ui_only = False
    """If true, setup() runs for WebUI jobs only, not for /sdapi requests"""

    sorting_priority = 0
    """Larger number will appear downwards in the UI."""

    param_order = None
    """Positional order of the ``params()`` names when it differs from their
    declaration (visual) order, e.g. the Sampler script's ``steps`` first."""

    script_id = None
    """The script's owner id in the Remi API (``seed``, ``sampler``, ...), set by the runner."""

    layout = None
    """The ``remi.ui.ScriptLayout`` built from ``params()``."""

    def title(self):
        """this function should return the title of the script. This is what will be displayed in the dropdown menu."""

        raise NotImplementedError()

    def params(self, is_img2img):
        """Return the script's parameters as one ``remi.ui`` layout node (or
        None for none). The inputs' depth-first declaration order (or
        ``param_order``) is the order of the positional arguments that run(),
        process() and the other hooks receive (A1111's ``ui()``, which
        built Gradio controls, is not supported)."""

        return None

    def show(self, is_img2img):
        """
        is_img2img is True if this function is called for the img2img interface, and False otherwise

        This function should return:
         - False if the script should not be shown in UI at all
         - True if the script should be shown in UI if it's selected in the scripts dropdown
         - script.AlwaysVisible if the script should be shown in UI at all times
        """

        return True

    def run(self, p, *args):
        """
        This function is called if the script has been selected in the script dropdown.
        It must do all processing and return the Processed object with results, same as
        one returned by processing.process_images.

        Usually the processing is done by calling the processing.process_images function.

        args are the script's params() values, in argument order
        """

        pass

    def setup(self, p, *args):
        """For AlwaysVisible scripts, this function is called when the processing object is set up, before any processing starts.
        args are the script's params() values, in argument order.
        """
        pass

    def before_process(self, p, *args):
        """
        This function is called very early during processing begins for AlwaysVisible scripts.
        You can modify the processing object (p) here, inject hooks, etc.
        args are the script's params() values, in argument order
        """

        pass

    def after_model_load(self, p, *args):
        """Run after checkpoint overrides have loaded the active model."""

        pass

    def process(self, p, *args):
        """
        This function is called before processing begins for AlwaysVisible scripts.
        You can modify the processing object (p) here, inject hooks, etc.
        args are the script's params() values, in argument order
        """

        pass

    def before_process_batch(self, p, *args, **kwargs):
        """
        Called before extra networks are parsed from the prompt, so you can add
        new extra network keywords to the prompt with this callback.

        **kwargs will have those items:
          - batch_number - index of current batch, from 0 to number of batches-1
          - prompts - list of prompts for current batch; you can change contents of this list but changing the number of entries will likely break things
          - seeds - list of seeds for current batch
          - subseeds - list of subseeds for current batch
        """

        pass

    def after_extra_networks_activate(self, p, *args, **kwargs):
        """
        Called after extra networks activation, before conds calculation
        allow modification of the network after extra networks activation been applied
        won't be call if p.disable_extra_networks

        **kwargs will have those items:
          - batch_number - index of current batch, from 0 to number of batches-1
          - prompts - list of prompts for current batch; you can change contents of this list but changing the number of entries will likely break things
          - seeds - list of seeds for current batch
          - subseeds - list of subseeds for current batch
          - extra_network_data - list of ExtraNetworkParams for current stage
        """
        pass

    def process_before_every_sampling(self, p, *args, **kwargs):
        """
        Similar to process(), called before every sampling.
        If you use high-res fix, this will be called two times.
        """
        pass

    def process_batch(self, p, *args, **kwargs):
        """
        Same as process(), but called for every batch.

        **kwargs will have those items:
          - batch_number - index of current batch, from 0 to number of batches-1
          - prompts - list of prompts for current batch; you can change contents of this list but changing the number of entries will likely break things
          - seeds - list of seeds for current batch
          - subseeds - list of subseeds for current batch
        """

        pass

    def postprocess_batch(self, p, *args, **kwargs):
        """
        Same as process_batch(), but called for every batch after it has been generated.

        **kwargs will have same items as process_batch, and also:
          - batch_number - index of current batch, from 0 to number of batches-1
          - images - torch tensor with all generated images, with values ranging from 0 to 1;
        """

        pass

    def postprocess_batch_list(self, p, pp: PostprocessBatchListArgs, *args, **kwargs):
        """
        Same as postprocess_batch(), but receives batch images as a list of 3D tensors instead of a 4D tensor.
        This is useful when you want to update the entire batch instead of individual images.

        You can modify the postprocessing object (pp) to update the images in the batch, remove images, add images, etc.
        If the number of images is different from the batch size when returning,
        then the script has the responsibility to also update the following attributes in the processing object (p):
          - p.prompts
          - p.negative_prompts
          - p.seeds
          - p.subseeds

        **kwargs will have same items as process_batch, and also:
          - batch_number - index of current batch, from 0 to number of batches-1
        """

        pass

    def on_mask_blend(self, p, mba: MaskBlendArgs, *args):
        """
        Called in inpainting mode when the original content is blended with the inpainted content.
        This is called at every step in the denoising process and once at the end.
        If is_final_blend is true, this is called for the final blending stage.
        Otherwise, denoiser and sigma are defined and may be used to inform the procedure.
        """

        pass

    def post_sample(self, p, ps: PostSampleArgs, *args):
        """
        Called after the samples have been generated,
        but before they have been decoded by the VAE, if applicable.
        Check getattr(samples, 'already_decoded', False) to test if the images are decoded.
        """

        pass

    def postprocess_image(self, p, pp: PostprocessImageArgs, *args):
        """
        Called for every image after it has been generated.
        """

        pass

    def postprocess_maskoverlay(self, p, ppmo: PostProcessMaskOverlayArgs, *args):
        """
        Called for every image after it has been generated.
        """

        pass

    def postprocess_image_after_composite(self, p, pp: PostprocessImageArgs, *args):
        """
        Called for every image after it has been generated.
        Same as postprocess_image but after inpaint_full_res composite
        So that it operates on the full image instead of the inpaint_full_res crop region.
        """

        pass

    def postprocess(self, p, processed, *args):
        """
        This function is called after processing ends for AlwaysVisible scripts.
        args are the script's params() values, in argument order
        """

        pass

    def cleanup(self, p, *args):
        """Release per-generation resources even when processing raises.

        Unlike ``postprocess``, this hook is guaranteed to run from
        ``process_images``' ``finally`` block. Implementations must be
        idempotent because successful generations call ``postprocess`` first.
        """

        pass

    def before_hr(self, p, *args):
        """
        This function is called before hires fix start.
        """
        pass


class ScriptBuiltinUI(Script):
    setup_for_ui_only = True

    def show(self, is_img2img):
        return AlwaysVisible


current_basedir = paths.script_path


def basedir():
    """returns the base directory for the current script. For scripts in the main scripts directory,
    this is the main directory (where webui.py resides), and for scripts in extensions directory
    (ie extensions/aesthetic/script/aesthetic.py), this is extension's directory (extensions/aesthetic)
    """
    return current_basedir


ScriptFile = namedtuple("ScriptFile", ["basedir", "filename", "path"])

scripts_data = []
postprocessing_scripts_data = []
scripts_not_loaded: list[tuple[ScriptFile, str]] = []
"""Script files that failed to import, with the reason (e.g. A1111 UI code
that imports a module the WebUI no longer ships); listed in bootstrap ``not_ported``."""
ScriptClassData = namedtuple("ScriptClassData", ["script_class", "path", "basedir", "module"])


@dataclass
class ScriptWithDependencies:
    script_canonical_name: str
    file: ScriptFile
    requires: list
    load_before: list
    load_after: list


def list_scripts(scriptdirname, extension, *, include_extensions=True):
    scripts = {}

    loaded_extensions = {ext.canonical_name: ext for ext in extensions.active()}
    loaded_extensions_scripts = {ext.canonical_name: [] for ext in extensions.active()}

    # build script dependency map
    root_script_basedir = os.path.join(paths.script_path, scriptdirname)
    if os.path.exists(root_script_basedir):
        for filename in sorted(os.listdir(root_script_basedir)):
            if not os.path.isfile(os.path.join(root_script_basedir, filename)):
                continue

            if os.path.splitext(filename)[1].lower() != extension:
                continue

            script_file = ScriptFile(paths.script_path, filename, os.path.join(root_script_basedir, filename))
            scripts[filename] = ScriptWithDependencies(filename, script_file, [], [], [])

    if include_extensions:
        for ext in extensions.active():
            extension_scripts_list = ext.list_files(scriptdirname, extension)
            for extension_script in extension_scripts_list:
                if not os.path.isfile(extension_script.path):
                    continue

                script_canonical_name = ("builtin/" if ext.is_builtin else "") + ext.canonical_name + "/" + extension_script.filename
                relative_path = scriptdirname + "/" + extension_script.filename

                script = ScriptWithDependencies(
                    script_canonical_name=script_canonical_name,
                    file=extension_script,
                    requires=ext.metadata.get_script_requirements("Requires", relative_path, scriptdirname),
                    load_before=ext.metadata.get_script_requirements("Before", relative_path, scriptdirname),
                    load_after=ext.metadata.get_script_requirements("After", relative_path, scriptdirname),
                )

                scripts[script_canonical_name] = script
                loaded_extensions_scripts[ext.canonical_name].append(script)

    for script_canonical_name, script in scripts.items():
        # load before requires inverse dependency
        # in this case, append the script name into the load_after list of the specified script
        for load_before in script.load_before:
            # if this requires an individual script to be loaded before
            other_script = scripts.get(load_before)
            if other_script:
                other_script.load_after.append(script_canonical_name)

            # if this requires an extension
            other_extension_scripts = loaded_extensions_scripts.get(load_before)
            if other_extension_scripts:
                for other_script in other_extension_scripts:
                    other_script.load_after.append(script_canonical_name)

        # if After mentions an extension, remove it and instead add all of its scripts
        for load_after in list(script.load_after):
            if load_after not in scripts and load_after in loaded_extensions_scripts:
                script.load_after.remove(load_after)

                for other_script in loaded_extensions_scripts.get(load_after, []):
                    script.load_after.append(other_script.script_canonical_name)

    dependencies = {}

    for script_canonical_name, script in scripts.items():
        for required_script in script.requires:
            if required_script not in scripts and required_script not in loaded_extensions:
                errors.report(f'Script "{script_canonical_name}" requires "{required_script}" to be loaded, but it is not.', exc_info=False)

        dependencies[script_canonical_name] = script.load_after

    ordered_scripts = topological_sort(dependencies)
    scripts_list = [scripts[script_canonical_name].file for script_canonical_name in ordered_scripts]

    return scripts_list


def list_files_with_name(filename):
    res = []

    dirs = [paths.script_path] + [ext.path for ext in extensions.active()]

    for dirpath in dirs:
        if not os.path.isdir(dirpath):
            continue

        path = os.path.join(dirpath, filename)
        if os.path.isfile(path):
            res.append(path)

    return res


def load_scripts():
    global current_basedir
    scripts_data.clear()
    postprocessing_scripts_data.clear()
    scripts_not_loaded.clear()
    script_callbacks.clear_callbacks()

    scripts_list = list_scripts("scripts", ".py") + list_scripts("modules/processing_scripts", ".py", include_extensions=False)

    for s in scripts_list:
        if s.basedir not in sys.path:
            sys.path = [s.basedir] + sys.path

    syspath = sys.path

    # print(f'Current System Paths = {syspath}')

    def register_scripts_from_module(module):
        for script_class in module.__dict__.values():
            if not inspect.isclass(script_class):
                continue

            if issubclass(script_class, Script):
                scripts_data.append(ScriptClassData(script_class, scriptfile.path, scriptfile.basedir, module))
            elif issubclass(script_class, scripts_postprocessing.ScriptPostprocessing):
                postprocessing_scripts_data.append(ScriptClassData(script_class, scriptfile.path, scriptfile.basedir, module))

    # here the scripts_list is already ordered
    # processing_script is not considered though
    for scriptfile in scripts_list:
        try:
            if scriptfile.basedir != paths.script_path:
                sys.path = [scriptfile.basedir] + sys.path
            current_basedir = scriptfile.basedir

            script_module = script_loading.load_module(scriptfile.path)
            register_scripts_from_module(script_module)

        except ModuleNotFoundError as error:
            scripts_not_loaded.append((scriptfile, f"needs the module {error.name!r}, which is not installed"))
            print(f"Script {scriptfile.path} is disabled: it needs the module {error.name!r}, which is not installed")

        except Exception as error:
            scripts_not_loaded.append((scriptfile, f"failed to load: {error}"))
            errors.report(f"Error loading script: {scriptfile.filename}", exc_info=True)

        finally:
            sys.path = syspath
            current_basedir = paths.script_path
            timer.startup_timer.record(scriptfile.filename)

    global scripts_txt2img, scripts_img2img, scripts_postproc

    scripts_txt2img = ScriptRunner()
    scripts_img2img = ScriptRunner()
    scripts_postproc = scripts_postprocessing.ScriptPostprocessingRunner()


def wrap_call(func, filename, funcname, *args, default=None, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception:
        errors.report(f"Error calling: {filename}/{funcname}", exc_info=True)

    return default


def has_params(script) -> bool:
    """Whether the script declares ``params()`` (the Remi parameter model)."""

    return getattr(type(script), "params", Script.params) is not Script.params


def legacy_ui_only(script) -> bool:
    """Whether the script only has A1111's ``ui()`` (Gradio controls, not
    supported) and no ``params()``: it cannot take arguments, so the runner
    leaves it out."""

    return callable(getattr(type(script), "ui", None)) and not has_params(script)


RESERVED_OWNERS = frozenset({"core", "settings"})
"""Owner ids of the Remi API that no script may take."""


def script_slug(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(title).lower()).strip("_")
    return slug if slug and not slug[0].isdigit() else f"s_{slug}"


def unique_id(base: str, taken) -> str:
    """``base``, or ``base_2``, ``base_3``, ... when taken or reserved."""

    candidate, number = base, 2
    while candidate in taken or candidate in RESERVED_OWNERS:
        candidate, number = f"{base}_{number}", number + 1
    return candidate


def ordered_ui_categories():
    """The txt2img/img2img parameter categories in the user's order
    (``opts.ui_reorder_list``), as the generation form lays them out."""

    from modules import shared_items

    user_order = {x.strip(): i * 2 + 1 for i, x in enumerate(getattr(shared.opts, "ui_reorder_list", []) or [])}
    for _, category in sorted(enumerate(shared_items.ui_reorder_categories()), key=lambda x: user_order.get(x[1], x[0] * 2 + 0)):
        yield category


class ScriptRunner:
    def __init__(self):
        self.scripts = []
        self.selectable_scripts = []
        self.alwayson_scripts = []
        self.titles = []
        self.title_map = {}

        self.callback_map = {}
        self.callback_names = [
            "setup",
            "before_process",
            "after_model_load",
            "process",
            "before_process_init_images",
            "before_process_batch",
            "after_extra_networks_activate",
            "process_batch",
            "process_before_every_sampling",
            "before_hr",
            "postprocess",
            "postprocess_batch",
            "postprocess_batch_list",
            "post_sample",
            "on_mask_blend",
            "postprocess_image",
            "postprocess_maskoverlay",
            "postprocess_image_after_composite",
            "cleanup",
        ]

        self.is_img2img = False
        self.by_id: dict[str, Script] = {}
        """scripts by ``script_id`` (the Remi API owner ids)"""
        self._slots_total = None
        self.disabled_scripts: list[tuple[Script, str]] = []
        """scripts left out (params() failed, or an A1111 ``ui()`` without params()), with the reason"""

    def initialize_scripts(self, is_img2img):
        from modules import scripts_auto_postprocessing

        self.is_img2img = is_img2img
        self._slots_total = None
        self.disabled_scripts = []
        self.scripts.clear()
        self.alwayson_scripts.clear()
        self.selectable_scripts.clear()

        auto_processing_scripts = scripts_auto_postprocessing.create_auto_preprocessing_script_data()

        for script_data in auto_processing_scripts + scripts_data:
            try:
                script = script_data.script_class()
            except Exception:
                errors.report(f"Error # failed to initialize Script {script_data.module}: ", exc_info=True)
                continue

            script.filename = script_data.path
            script.is_txt2img = not is_img2img
            script.is_img2img = is_img2img
            script.tabname = "img2img" if is_img2img else "txt2img"

            visibility = script.show(script.is_img2img)

            if visibility == AlwaysVisible:
                self.scripts.append(script)
                self.alwayson_scripts.append(script)
                script.alwayson = True

            elif visibility:
                self.scripts.append(script)
                self.selectable_scripts.append(script)

        self.callback_map.clear()

        self.build_layouts()

    # region Remi parameter model (params() layouts, named values)

    def build_layouts(self):
        """Give every script its ``script_id`` and its ``ScriptLayout`` from
        ``params()`` (none: no arguments). A script that only has A1111's
        ``ui()`` is disabled."""

        from remi.ui import ScriptLayout

        self.by_id = {}
        for script in list(self.scripts):
            title = wrap_call(script.title, script.filename, "title") or os.path.basename(script.filename or "script")
            script.name = title.lower()
            script.script_id = unique_id(script_slug(title), self.by_id)
            self.by_id[script.script_id] = script
            script.layout = None

            if legacy_ui_only(script):
                print(f"Script {script.filename} is disabled: it declares ui() (Gradio controls) instead of params()")
                self.disable_script(script, "declares ui() instead of params()")
                continue
            try:
                script.layout = ScriptLayout(script.params(script.is_img2img), owner=script.script_id, order=script.param_order)
            except Exception as error:
                errors.report(f"Error in params() of {script.filename}; the script is disabled", exc_info=True)
                self.disable_script(script, f"params() failed: {error}")
                continue
            script.api_info = self._layout_api_info(script)

        self._index_titles()

    def _index_titles(self):
        self.titles = [wrap_call(script.title, script.filename, "title") or f"{script.filename} [error]" for script in self.selectable_scripts]
        self.title_map = {script.name: script for script in self.scripts}

    def disable_script(self, script, reason: str):
        """Leave a script whose parameters cannot be read out of this runner:
        no slots, no schema entry, no processing hooks."""

        for collection in (self.scripts, self.alwayson_scripts, self.selectable_scripts):
            if script in collection:
                collection.remove(script)
        if self.by_id.get(script.script_id) is script:
            del self.by_id[script.script_id]
        self.disabled_scripts.append((script, reason))
        self.callback_map.clear()
        self._slots_total = None
        self._index_titles()

    def _layout_api_info(self, script):
        try:
            import modules.api.models as api_models
            from remi.ui import Context, nodes

            ctx = Context()
            args = []
            for node in script.layout.inputs:
                arg = api_models.ScriptArg(label=node.label or node.name, value=node.resolve_default(ctx))
                for field in ("minimum", "maximum", "step"):
                    if getattr(node, field, None) is not None:
                        setattr(arg, field, getattr(node, field))
                if isinstance(node, (nodes.Choice, nodes.MultiChoice)):
                    arg.choices = [str(value) for _label, value in node.resolve_choices(ctx)]
                args.append(arg)
            return api_models.ScriptInfo(name=script.name, is_img2img=script.is_img2img, is_alwayson=script.alwayson, args=args)
        except Exception:
            errors.report(f"Error building API info for {script.filename}", exc_info=True)
            return None

    def layout_for(self, script) -> "ScriptLayout":
        """The script's layout (``params()``)."""

        return script.layout

    def slot_order(self) -> list:
        """The order in which scripts get their argument slots (by UI
        category, then sorting priority; the A1111 order, which positional
        ``/sdapi`` clients and saved jobs rely on)."""

        alwayson = sorted(self.alwayson_scripts, key=lambda x: x.sorting_priority)
        selectable = sorted(self.selectable_scripts, key=lambda x: x.sorting_priority)
        order = []
        for category in ordered_ui_categories():
            if category == "scripts":
                order += [script for script in alwayson if script.section is None] + selectable
            order += [script for script in alwayson if script.section == category]
        return order + [script for script in self.scripts if script not in order]

    def ensure_arg_slots(self) -> int:
        """Assign ``args_from``/``args_to`` from the layouts; returns the
        length of the full script_args list."""

        if self._slots_total is None:
            position = 1  # slot 0 is the selected script's index
            for script in self.slot_order():
                count = len(self.layout_for(script))
                script.args_from, script.args_to = position, position + count
                position += count
            self._slots_total = position
        return self._slots_total

    def selector_choices(self) -> list[str]:
        return ["None", *self.titles]

    def selector_index(self, title) -> int:
        """Slot 0's value for the selected script's title (0: none)."""

        from remi.ui import ParamError, ValidationError

        if title in (None, "", "None"):
            return 0
        if isinstance(title, int) and not isinstance(title, bool) and 0 <= title <= len(self.titles):
            return title
        if title in self.titles:
            return self.titles.index(title) + 1
        raise ValidationError([ParamError(f"unknown script {title!r}", owner="core", param="script")])

    def script_args_from_values(self, selected, values: dict | None, ctx: "Context | None" = None, *, unknown: str = "error") -> list:
        """The full ``p.script_args`` for named values ``{script_id: {param:
        value}}`` (or a list of a script's values in argument order) and the
        selected script's title. Missing params take their defaults; the
        unselected selectable scripts get their defaults."""

        from remi.ui import Context, ParamError, ValidationError

        ctx = ctx or Context()
        if values is None:
            values = {}
        elif not isinstance(values, Mapping):
            raise ValidationError([ParamError(f"scripts: expected an object of values by script id, got {type(values).__name__}")])
        problems = []
        if unknown == "error":
            problems += [ParamError("unknown script", owner=owner) for owner in values if owner not in self.by_id]

        index = self.selector_index(selected)
        chosen = self.selectable_scripts[index - 1] if index else None
        args = [None] * self.ensure_arg_slots()
        args[0] = index
        for script in self.scripts:
            if script.args_from is None or script.args_to is None:
                continue
            layout = self.layout_for(script)
            owner_values = None if (not script.alwayson and script is not chosen) else values.get(script.script_id)
            if isinstance(owner_values, (list, tuple)):
                if len(owner_values) > len(layout.names):
                    problems.append(ParamError(f"{len(owner_values)} values for {len(layout.names)} params", owner=script.script_id))
                    continue
                owner_values = dict(zip(layout.names, owner_values))
            try:
                positional = layout.values_to_args(owner_values, ctx, unknown=unknown)
            except ValidationError as error:
                problems.append(error)
                continue
            if len(positional) != script.args_to - script.args_from:
                raise RuntimeError(f"{script.filename}: params() declares {len(positional)} arguments, {script.args_to - script.args_from} slots were assigned")
            args[script.args_from : script.args_to] = positional
        if problems:
            raise ValidationError.merge(problems)
        return args

    def script_values_from_args(self, args, ctx: "Context | None" = None) -> tuple:
        """Inverse of :meth:`script_args_from_values`: ``(selected title,
        {script_id: values})`` for a full script_args list."""

        from remi.ui import Context

        ctx = ctx or Context()
        self.ensure_arg_slots()
        args = list(args)
        index = args[0] if args and isinstance(args[0], int) else 0
        selected = self.titles[index - 1] if 0 < index <= len(self.titles) else "None"
        values = {}
        for script in self.scripts:
            if script.args_from is None or script.args_to is None:
                continue
            values[script.script_id] = self.layout_for(script).args_to_values(args[script.args_from : script.args_to], ctx)
        return selected, values

    def scripts_schema(self, ctx: "Context | None" = None) -> list["ScriptEntry"]:
        """The ``scripts`` part of ``GET /schema/{tab}`` (schema-format.md §4)."""

        from remi.ui import Context, jsonable

        ctx = ctx or Context()
        self.ensure_arg_slots()
        result = []
        ordered = sorted(self.alwayson_scripts, key=lambda x: x.sorting_priority) + sorted(self.selectable_scripts, key=lambda x: x.sorting_priority)
        for script in ordered:
            layout = self.layout_for(script)
            title = wrap_call(script.title, script.filename, "title") or script.name
            entry = {
                "id": script.script_id,
                "title": title,
                "kind": "alwayson" if script.alwayson else "selectable",
                # Selectable scripts always render in the scripts area
                "section": (script.section if isinstance(script.section, str) else None) if script.alwayson else "scripts",
                "order": script.sorting_priority,
            }
            if not script.alwayson:
                entry["visible_if"] = {"param": "core.script", "eq": title}
            entry.update(layout=layout.schema(ctx), defaults=jsonable(layout.defaults(ctx)), args=layout.names)
            result.append(entry)
        return result

    def paste_values(self, params: dict, ctx: "Context | None" = None) -> tuple[dict, list]:
        """Values every script reads from parsed infotext ``params``:
        ``({script_id: values}, problems)``. Owners with nothing pasted are left out."""

        from remi.ui import Context

        ctx = ctx or Context()
        values, problems = {}, []
        for script in self.scripts:
            try:
                layout = self.layout_for(script)
            except Exception:
                errors.report(f"Error reading the params of {script.filename}", exc_info=True)
                continue
            pasted, failed = layout.paste(params, ctx)
            if pasted:
                values[script.script_id] = pasted
            problems += failed
        return values, problems

    def preset_values(self, label: str) -> dict[str, dict]:
        """``{script_id: {param: value}}`` the Anima preset ``label`` gives
        the ported scripts' bound params (see ``remi.core.presets`` for the
        patch with the core form and the conflict-control unlocks)."""

        from modules.anima_presets import preset_named_values
        from remi.core.presets import preset_bindings

        layouts = {script.script_id: script.layout for script in self.scripts if script.layout is not None}
        return preset_named_values(label, preset_bindings(layouts))

    def preset_bindings(self) -> dict[str, list[tuple[str, str]]]:
        """Anima preset field -> [(script_id, param)] over the ported scripts."""

        bindings: dict[str, list[tuple[str, str]]] = {}
        for script in self.scripts:
            if script.layout is not None:
                for key, (name, _value) in script.layout.presets.items():
                    bindings.setdefault(key, []).append((script.script_id, name))
        return bindings

    # endregion

    def run(self, p, *args):
        script_index = args[0]

        if script_index == 0 or script_index is None:
            return None

        script = self.selectable_scripts[script_index - 1]

        if script is None:
            return None

        script_args = args[script.args_from : script.args_to]
        processed = script.run(p, *script_args)

        shared.total_tqdm.clear()

        return processed

    def list_scripts_for_method(self, method_name):
        return self.alwayson_scripts

    def create_ordered_callbacks_list(self, method_name, *, enable_user_sort=True):
        script_list = self.list_scripts_for_method(method_name)
        category = f"script_{method_name}"
        callbacks = []

        for script in script_list:
            if getattr(script.__class__, method_name, None) == getattr(Script, method_name, None):
                continue

            script_callbacks.add_callback(callbacks, script, category=category, name=script.__class__.__name__, filename=script.filename)

        return script_callbacks.sort_callbacks(category, callbacks, enable_user_sort=enable_user_sort)

    def ordered_callbacks(self, method_name, *, enable_user_sort=True):
        script_list = self.list_scripts_for_method(method_name)
        category = f"script_{method_name}"

        scripts_len, callbacks = self.callback_map.get(category, (-1, None))

        if callbacks is None or scripts_len != len(script_list):
            callbacks = self.create_ordered_callbacks_list(method_name, enable_user_sort=enable_user_sort)
            self.callback_map[category] = len(script_list), callbacks

        return callbacks

    def ordered_scripts(self, method_name):
        return [x.callback for x in self.ordered_callbacks(method_name)]

    def before_process(self, p):
        for script in self.ordered_scripts("before_process"):
            try:
                script_args = p.script_args[script.args_from : script.args_to]
                script.before_process(p, *script_args)
            except Exception:
                errors.report(f"Error running before_process: {script.filename}", exc_info=True)

    def after_model_load(self, p):
        for script in self.ordered_scripts("after_model_load"):
            try:
                script_args = p.script_args[script.args_from : script.args_to]
                script.after_model_load(p, *script_args)
            except Exception:
                errors.report(f"Error running after_model_load: {script.filename}", exc_info=True)

    def process(self, p):
        for script in self.ordered_scripts("process"):
            try:
                script_args = p.script_args[script.args_from : script.args_to]
                script.process(p, *script_args)
            except Exception:
                errors.report(f"Error running process: {script.filename}", exc_info=True)

    def process_before_every_sampling(self, p, **kwargs):
        for script in self.ordered_scripts("process_before_every_sampling"):
            try:
                script_args = p.script_args[script.args_from : script.args_to]
                script.process_before_every_sampling(p, *script_args, **kwargs)
            except Exception:
                errors.report(f"Error running process_before_every_sampling: {script.filename}", exc_info=True)

    def before_process_batch(self, p, **kwargs):
        for script in self.ordered_scripts("before_process_batch"):
            try:
                script_args = p.script_args[script.args_from : script.args_to]
                script.before_process_batch(p, *script_args, **kwargs)
            except ScriptAbort:
                raise
            except Exception:
                errors.report(f"Error running before_process_batch: {script.filename}", exc_info=True)

    def before_process_init_images(self, p, pp, **kwargs):
        for script in self.ordered_scripts("before_process_init_images"):
            try:
                script_args = p.script_args[script.args_from : script.args_to]
                script.before_process_init_images(p, pp, *script_args, **kwargs)
            except Exception:
                errors.report(f"Error running before_process_init_images: {script.filename}", exc_info=True)

    def after_extra_networks_activate(self, p, **kwargs):
        for script in self.ordered_scripts("after_extra_networks_activate"):
            try:
                script_args = p.script_args[script.args_from : script.args_to]
                script.after_extra_networks_activate(p, *script_args, **kwargs)
            except Exception:
                errors.report(f"Error running after_extra_networks_activate: {script.filename}", exc_info=True)

    def process_batch(self, p, **kwargs):
        for script in self.ordered_scripts("process_batch"):
            try:
                script_args = p.script_args[script.args_from : script.args_to]
                script.process_batch(p, *script_args, **kwargs)
            except Exception:
                errors.report(f"Error running process_batch: {script.filename}", exc_info=True)

    def postprocess(self, p, processed):
        for script in self.ordered_scripts("postprocess"):
            try:
                script_args = p.script_args[script.args_from : script.args_to]
                script.postprocess(p, processed, *script_args)
            except Exception:
                errors.report(f"Error running postprocess: {script.filename}", exc_info=True)

    def cleanup(self, p):
        for script in self.ordered_scripts("cleanup"):
            try:
                script_args = p.script_args[script.args_from : script.args_to]
                script.cleanup(p, *script_args)
            except Exception:
                errors.report(f"Error running cleanup: {script.filename}", exc_info=True)

    def postprocess_batch(self, p, images, **kwargs):
        for script in self.ordered_scripts("postprocess_batch"):
            try:
                script_args = p.script_args[script.args_from : script.args_to]
                script.postprocess_batch(p, *script_args, images=images, **kwargs)
            except Exception:
                errors.report(f"Error running postprocess_batch: {script.filename}", exc_info=True)

    def postprocess_batch_list(self, p, pp: PostprocessBatchListArgs, **kwargs):
        for script in self.ordered_scripts("postprocess_batch_list"):
            try:
                script_args = p.script_args[script.args_from : script.args_to]
                script.postprocess_batch_list(p, pp, *script_args, **kwargs)
            except Exception:
                errors.report(f"Error running postprocess_batch_list: {script.filename}", exc_info=True)

    def post_sample(self, p, ps: PostSampleArgs):
        for script in self.ordered_scripts("post_sample"):
            try:
                script_args = p.script_args[script.args_from : script.args_to]
                script.post_sample(p, ps, *script_args)
            except Exception:
                errors.report(f"Error running post_sample: {script.filename}", exc_info=True)

    def on_mask_blend(self, p, mba: MaskBlendArgs):
        for script in self.ordered_scripts("on_mask_blend"):
            try:
                script_args = p.script_args[script.args_from : script.args_to]
                script.on_mask_blend(p, mba, *script_args)
            except Exception:
                errors.report(f"Error running post_sample: {script.filename}", exc_info=True)

    def postprocess_image(self, p, pp: PostprocessImageArgs):
        for script in self.ordered_scripts("postprocess_image"):
            try:
                script_args = p.script_args[script.args_from : script.args_to]
                script.postprocess_image(p, pp, *script_args)
            except Exception:
                errors.report(f"Error running postprocess_image: {script.filename}", exc_info=True)

    def postprocess_maskoverlay(self, p, ppmo: PostProcessMaskOverlayArgs):
        for script in self.ordered_scripts("postprocess_maskoverlay"):
            try:
                script_args = p.script_args[script.args_from : script.args_to]
                script.postprocess_maskoverlay(p, ppmo, *script_args)
            except Exception:
                errors.report(f"Error running postprocess_image: {script.filename}", exc_info=True)

    def postprocess_image_after_composite(self, p, pp: PostprocessImageArgs):
        for script in self.ordered_scripts("postprocess_image_after_composite"):
            try:
                script_args = p.script_args[script.args_from : script.args_to]
                script.postprocess_image_after_composite(p, pp, *script_args)
            except Exception:
                errors.report(f"Error running postprocess_image_after_composite: {script.filename}", exc_info=True)

    def script(self, title):
        return self.title_map.get(title.lower())

    def reload_sources(self, cache):
        for si, script in list(enumerate(self.scripts)):
            args_from = script.args_from
            args_to = script.args_to
            filename = script.filename

            module = cache.get(filename, None)
            if module is None:
                module = script_loading.load_module(script.filename)
                cache[filename] = module

            for script_class in module.__dict__.values():
                if type(script_class) == type and issubclass(script_class, Script):
                    self.scripts[si] = script_class()
                    self.scripts[si].filename = filename
                    self.scripts[si].args_from = args_from
                    self.scripts[si].args_to = args_to
                    # Body-only reload: the parameter model is kept as it was
                    self.scripts[si].script_id = script.script_id
                    self.scripts[si].layout = script.layout

    def before_hr(self, p):
        for script in self.ordered_scripts("before_hr"):
            try:
                script_args = p.script_args[script.args_from : script.args_to]
                script.before_hr(p, *script_args)
            except Exception:
                errors.report(f"Error running before_hr: {script.filename}", exc_info=True)

    def setup_scripts(self, p, *, is_ui=True):
        for script in self.ordered_scripts("setup"):
            if not is_ui and script.setup_for_ui_only:
                continue

            try:
                script_args = p.script_args[script.args_from : script.args_to]
                script.setup(p, *script_args)
            except Exception:
                errors.report(f"Error running setup: {script.filename}", exc_info=True)

    def set_named_arg(self, args, script_name, arg_elem_id, value, fuzzy=False):
        """Set one argument of script ``script_name`` (its lowercased title)
        in ``args`` (``p.script_args``): ``arg_elem_id`` is the param name
        (``fuzzy``: a substring of it). Returns the updated args; raises
        RuntimeError for an unknown script or param."""

        script = next((x for x in self.scripts if x.name == script_name), None)
        if script is None:
            raise RuntimeError(f"script {script_name} not found")

        self.ensure_arg_slots()
        for i, name in enumerate(self.layout_for(script).names):
            if arg_elem_id in name if fuzzy else arg_elem_id == name:
                index = script.args_from + i

                if isinstance(args, tuple):
                    return args[:index] + (value,) + args[index + 1 :]
                elif isinstance(args, list):
                    args[index] = value
                    return args
                else:
                    raise RuntimeError(f"args is not a list or tuple, but {type(args)}")
        raise RuntimeError(f"param {arg_elem_id} not found in script {script_name}")


scripts_txt2img: ScriptRunner = None
scripts_img2img: ScriptRunner = None
scripts_postproc: scripts_postprocessing.ScriptPostprocessingRunner = None


def reload_script_body_only():
    cache = {}
    scripts_txt2img.reload_sources(cache)
    scripts_img2img.reload_sources(cache)


reload_scripts = load_scripts  # compatibility alias
