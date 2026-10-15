import dataclasses

from modules import shared


@dataclasses.dataclass
class PostprocessedImageSharedInfo:
    target_width: int = None
    target_height: int = None


class PostprocessedImage:
    def __init__(self, image):
        self.image = image
        self.info = {}
        self.shared = PostprocessedImageSharedInfo()
        self.extra_images = []
        self.nametags = []
        self.disable_processing = False
        self.caption = None

    def get_suffix(self, used_suffixes=None):
        used_suffixes = {} if used_suffixes is None else used_suffixes
        suffix = "-".join(self.nametags)
        if suffix:
            suffix = "-" + suffix

        if suffix not in used_suffixes:
            used_suffixes[suffix] = 1
            return suffix

        for i in range(1, 100):
            proposed_suffix = suffix + "-" + str(i)

            if proposed_suffix not in used_suffixes:
                used_suffixes[proposed_suffix] = 1
                return proposed_suffix

        return suffix

    def create_copy(self, new_image, *, nametags=None, disable_processing=False):
        pp = PostprocessedImage(new_image)
        pp.shared = self.shared
        pp.nametags = self.nametags.copy()
        pp.info = self.info.copy()
        pp.disable_processing = disable_processing

        if nametags is not None:
            pp.nametags += nametags

        return pp


class ScriptPostprocessing:
    filename = None
    arg_names = None
    """The ``params()`` names, in argument order (set by ``remi.core.extras.assign_args``)."""
    args_from = None
    args_to = None

    order = 1000
    """scripts will be ordred by this value in the Extras form"""

    name = None
    """this function should return the title of the script."""

    def params(self):
        """The operation's parameters as one ``remi.ui`` layout node (None:
        none); process() receives them by name."""

        return None

    def process(self, pp: PostprocessedImage, **args):
        """
        This function is called to postprocess the image.
        args are the params() values by name
        """

        pass

    def process_firstpass(self, pp: PostprocessedImage, **args):
        """
        Called for all scripts before calling process(). Scripts can examine the image here and set fields
        of the pp object to communicate things to other scripts.
        args are the params() values by name
        """

        pass

    def image_changed(self):
        pass


class ScriptPostprocessingRunner:
    def __init__(self):
        self.scripts = None
        self.disabled_scripts = []
        """operations left out (params() failed, or an A1111 ``ui()`` without params()), with the reason"""

    def initialize_scripts(self, scripts_data):
        self.scripts = []
        self.disabled_scripts = []

        for script_data in scripts_data:
            script: ScriptPostprocessing = script_data.script_class()
            script.filename = script_data.path

            if script.name == "Simple Upscale":
                continue

            self.scripts.append(script)

    def scripts_in_preferred_order(self):
        if self.scripts is None:
            import modules.scripts
            self.initialize_scripts(modules.scripts.postprocessing_scripts_data)

        scripts_order = shared.opts.postprocessing_operation_order
        scripts_filter_out = set(shared.opts.postprocessing_disable_in_extras)

        def script_score(name):
            for i, possible_match in enumerate(scripts_order):
                if possible_match == name:
                    return i

            return len(self.scripts)

        filtered_scripts = [script for script in self.scripts if script.name not in scripts_filter_out]
        script_scores = {script.name: (script_score(script.name), script.order, script.name, original_index) for original_index, script in enumerate(filtered_scripts)}

        return sorted(filtered_scripts, key=lambda x: script_scores[x.name])

    def run(self, pp: PostprocessedImage, args):
        scripts = []

        for script in self.scripts_in_preferred_order():
            script_args = args[script.args_from:script.args_to]

            process_args = dict(zip(script.arg_names or (), script_args))

            scripts.append((script, process_args))

        for script, process_args in scripts:
            script.process_firstpass(pp, **process_args)

        all_images = [pp]

        for script, process_args in scripts:
            if shared.state.skipped:
                break

            shared.state.job = script.name

            for single_image in all_images.copy():

                if not single_image.disable_processing:
                    script.process(single_image, **process_args)

                for extra_image in single_image.extra_images:
                    if not isinstance(extra_image, PostprocessedImage):
                        extra_image = single_image.create_copy(extra_image)

                    all_images.append(extra_image)

                single_image.extra_images.clear()

        pp.extra_images = all_images[1:]

    def create_args_for_run(self, scripts_args):
        """Positional args for :meth:`run` from ``{operation name: {param:
        value}}`` (the A1111 extras API); every other arg is None (an
        operation without values stays off)."""

        from remi.core.extras import assign_args

        assigned = assign_args(self)
        args = [None] * max((script.args_to for script, _layout in assigned), default=0)

        for script, layout in assigned:
            if (values := scripts_args.get(script.name)) is not None:
                args[script.args_from : script.args_to] = [values.get(name) for name in layout.names]

        return args

    def image_changed(self):
        for script in self.scripts_in_preferred_order():
            script.image_changed()

