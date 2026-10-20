"""The "Checkpoint Merger" page (A1111's ``modelmerger`` tab), a
``remi.pages`` page registered by ``scripts/checkpoint_merger.py``.

It keeps the former tab's controls in the same order: "Save Current Model"
(UNet only, or UNet + CLIP + VAE), the models A / B / C with a refresh
button, the interpolation method and multiplier, the output filename and the
layers to discard, the metadata options with their preview, and Merge. The
merge (``modules.extras.run_modelmerger``) and the saves are page tasks:
they run behind generation (holding the generation lock, as the Gradio
button's GPU call did), report progress and can be cancelled. The new file
shows up in the checkpoint lists right away (``networks.changed``).
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from remi import pages, ui
from remi.ui.conditions import param

PAGE_ID = "modelmerger"
TITLE = "Checkpoint Merger"

NO_INTERPOLATION = "No Interpolation"
WEIGHTED_SUM = "Weighted Sum"
ADD_DIFFERENCE = "Add Difference"
METHODS = (NO_INTERPOLATION, WEIGHTED_SUM, ADD_DIFFERENCE)
"""``modules.extras.InterpolationMethod`` titles (kept here so the layout
needs no torch import)."""
METHOD_NOTES = {
    NO_INTERPOLATION: "Requires 1 model; mainly for format conversion",
    WEIGHTED_SUM: "Requires 2 models; the result is A × (1 − M) + B × M",
    ADD_DIFFERENCE: "Requires 3 models; the result is A + (B − C) × M",
}
STAGES = ("Loading A", "Loading B", "Loading C", "Merging A and B", "Merging B and C", "Saving")
"""Progress texts of ``run_modelmerger`` (catalog keys)."""
NO_MODEL = ""


def _checkpoints() -> list:
    """The checkpoints, labelled by title; the value is the file name, which
    stays the same when the title gains its hash (both are aliases the merge
    looks models up by)."""

    from modules import sd_models

    infos = sorted(sd_models.checkpoints_list.values(), key=lambda info: info.title)
    return [("Select a model", NO_MODEL), *((info.title, info.name) for info in infos)]


def _changed_checkpoints() -> None:
    """The checkpoint folder has a new file: re-list it and tell the clients."""

    from modules import sd_models
    from remi.networks import notify_changed

    sd_models.list_models()
    notify_changed("checkpoints")


def _problem(values: dict[str, Any]) -> str | None:
    """Why the merge cannot start (a catalog key), or None."""

    method = values["interp_method"]
    if not values["primary_model_name"]:
        return "Choose the primary model (A)"
    if method != NO_INTERPOLATION and not values["secondary_model_name"]:
        return "Choose the secondary model (B)"
    if method == ADD_DIFFERENCE and not values["tertiary_model_name"]:
        return "Choose the tertiary model (C)"
    if values["discard_weights"]:
        try:
            re.compile(values["discard_weights"])
        except re.error:
            return "Layers to Discard is not a valid regular expression"
    return None


def _inputs(values: dict[str, Any]) -> list[str]:
    """The checkpoints the merge reads (A, then B and C when the method uses them)."""

    method = values["interp_method"]
    names = [values["primary_model_name"]]
    if method != NO_INTERPOLATION:
        names.append(values["secondary_model_name"])
    if method == ADD_DIFFERENCE:
        names.append(values["tertiary_model_name"])
    return names


def _overwrites_input(values: dict[str, Any]) -> bool:
    """Whether the chosen filename is one of the merged checkpoints (never
    overwrite a model while reading it)."""

    from modules import extras, sd_models

    if not values["custom_name"].strip():
        return False
    target = extras.merge_output_path(values["custom_name"])
    for name in _inputs(values):
        info = sd_models.checkpoint_aliases.get(name)
        if info is not None and os.path.realpath(info.filename) == target:
            return True
    return False


class CheckpointMerger(pages.Page):
    def params(self):
        uses_b = param("interp_method", ne=NO_INTERPOLATION)
        return ui.Column(
            ui.Section(
                "Save Current Model (only supports SD1 / SDXL / Flux)",
                open=False,
                children=[
                    ui.Row(
                        ui.Text("save_filename", "flux.safetensors", label="Filename", info='saved in the "Stable-diffusion" folder', width="fill", persist=False),
                        ui.Custom("save_unet", "task-runner", None, props={"action": "save_unet", "label": "Save UNet Only", "variant": "secondary"}, label="", persist=False, submit=False),
                        ui.Custom("save_checkpoint", "task-runner", None, props={"action": "save_checkpoint", "label": "Save Checkpoint (UNet + Clip + VAE)", "variant": "secondary"}, label="", persist=False, submit=False),
                    ),
                    ui.Output("save_result", kind="text"),
                ],
            ),
            ui.Row(
                ui.Choice("primary_model_name", _checkpoints, NO_MODEL, label="Primary Model (A)", width="fill", persist=False),
                ui.Choice("secondary_model_name", _checkpoints, NO_MODEL, label="Secondary Model (B)", width="fill", visible_if=uses_b, persist=False),
                ui.Choice("tertiary_model_name", _checkpoints, NO_MODEL, label="Tertiary Model (C)", width="fill", visible_if=param("interp_method", eq=ADD_DIFFERENCE), persist=False),
                ui.Button("Refresh", "refresh", "tool"),
            ),
            ui.Row(
                ui.Choice("interp_method", list(METHODS), WEIGHTED_SUM, style="radio", label="Interpolation Method", persist=False),
                ui.Slider("multiplier", 0.0, 1.0, 0.5, 0.05, label="Multiplier (M)", visible_if=uses_b, persist=False),
            ),
            *(ui.Markdown(note, visible_if=param("interp_method", eq=method)) for method, note in METHOD_NOTES.items()),
            ui.Row(
                ui.Text("custom_name", "", label="Filename to Save", info="based on the interpolation method when empty", persist=False),
                ui.Text("discard_weights", "", label="Layers to Discard", info="in regex format", persist=False),
            ),
            ui.Section(
                "Save Metadata",
                enable=ui.Toggle("save_metadata", True, label="Save Metadata", persist=False),
                open=True,
                children=[
                    ui.Row(
                        ui.MultiChoice("config_source", ["A", "B", "C"], ["A", "B", "C"], style="checkboxes", label="Copy Metadata from", persist=False),
                        ui.Toggle("add_merge_recipe", True, label="Include Merge Recipe", persist=False),
                    ),
                    ui.Button("Preview Metadata from Models", "read_metadata"),
                    ui.Output("metadata_json", kind="json", label="Metadata in JSON Format"),
                ],
            ),
            # The action and its outcome in one framed group, like the panels above
            ui.Group(
                ui.Custom("merge", "task-runner", None, props={"action": "merge", "label": "Merge"}, label="", persist=False, submit=False),
                ui.Output("result", kind="text", label="Result"),
            ),
        )

    @ui.action("refresh")
    def refresh(self, ctx, values):
        _changed_checkpoints()
        choices = _checkpoints()
        return {name: {"choices": choices} for name in ("primary_model_name", "secondary_model_name", "tertiary_model_name")}

    @ui.action("read_metadata")
    def read_metadata(self, ctx, values):
        from modules import extras

        resolved = self.values(ctx, values, uploads=False)
        metadata = extras.read_metadata(resolved["primary_model_name"], resolved["secondary_model_name"], resolved["tertiary_model_name"])
        if not metadata:
            return {"__outputs__": {"metadata_json": None}, **ui.toast("Choose a model first", "warning")}
        return {"__outputs__": {"metadata_json": json.loads(json.dumps(metadata, default=str))}}

    @ui.action("merge")
    def merge(self, ctx, values):
        from modules import extras

        resolved = self.values(ctx, values, uploads=False)
        problem = _problem(resolved)
        if problem is None:
            try:
                if _overwrites_input(resolved):
                    problem = "The filename is one of the models being merged; choose another"
            except ValueError:
                problem = "The filename must stay inside the checkpoint folder"
        if problem is not None:
            return ui.toast(problem, "warning")

        def work(task: pages.Task) -> dict[str, Any]:
            def progress(text: str, fraction: float | None) -> None:
                task.check()  # Cancel stops the merge between two steps
                task.progress(fraction, text)

            result = extras.run_modelmerger(
                None,
                resolved["primary_model_name"],
                resolved["secondary_model_name"],
                resolved["tertiary_model_name"],
                resolved["interp_method"],
                float(resolved["multiplier"]),
                resolved["custom_name"].strip(),
                resolved["discard_weights"],
                bool(resolved["save_metadata"]),
                list(resolved["config_source"]),
                bool(resolved["add_merge_recipe"]),
                progress=progress,
            )
            if result.startswith("Failed"):
                return {"__outputs__": {"result": result}, **ui.toast(ui.message("Merge failed: {{error}}", error=result), "error")}
            path = result.removeprefix("Checkpoint saved to ")
            _changed_checkpoints()
            return {"__outputs__": {"result": ui.message("Checkpoint saved to {{path}}", path=path)}, **ui.toast("Checkpoint merged", "success")}

        return self.start_task(ctx, "merge", work, gpu=True)

    def _save(self, ctx, values, *, unet: bool):
        from modules import extras

        resolved = self.values(ctx, values, uploads=False)
        try:
            path = extras.merge_output_path(resolved["save_filename"])
        except ValueError:
            return ui.toast("The filename must stay inside the checkpoint folder", "warning")

        def work(task: pages.Task) -> dict[str, Any]:
            from modules import sd_models, shared

            task.progress(None, "Loading the model…")
            sd_models.forge_model_reload()
            task.check()
            task.progress(None, "Saving…")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            saved = shared.sd_model.save_unet(path) if unet else shared.sd_model.save_checkpoint(path)
            _changed_checkpoints()
            text = "UNet saved to {{path}}" if unet else "Checkpoint saved to {{path}}"
            return {"__outputs__": {"save_result": ui.message(text, path=str(saved))}, **ui.toast("Saved", "success")}

        return self.start_task(ctx, "save_unet" if unet else "save_checkpoint", work, gpu=True)

    @ui.action("save_unet")
    def save_unet(self, ctx, values):
        return self._save(ctx, values, unet=True)

    @ui.action("save_checkpoint")
    def save_checkpoint(self, ctx, values):
        return self._save(ctx, values, unet=False)


def register() -> CheckpointMerger:
    """Add the page (again after a "Reload UI": its script registers it)."""

    return pages.register(CheckpointMerger(PAGE_ID, TITLE, "git-merge"))
