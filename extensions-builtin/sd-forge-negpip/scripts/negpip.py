import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.processing import StableDiffusionProcessing

import torch
from backend.logging import setup_logger
from lib_negpip import IS_NEO, INCOMPATIBLE_EXTENSIONS
from lib_negpip.anima import patch_anima_negpip, patched_engine
from lib_negpip.sd import patch_sd_negpip
from lib_negpip.utils import any_negative, any_negative_target, hr_dealer, negative_weights, reset_prompt_cache

from modules import devices, scripts
from modules.anima_feature_conflicts import record_conflict_resolution
from modules.anima_support import is_anima_engine, schedule_index
from modules.prompt_parser import (
    SdConditioning,
    get_learned_conditioning,
    get_learned_conditioning_prompt_schedules,
)
from modules.script_callbacks import CFGDenoiserParams, on_cfg_denoiser

logger = logging.getLogger(__name__)
setup_logger(logger)


def _verify_ext(p: "StableDiffusionProcessing"):
    for ext in p.scripts.scripts:
        if ext.title() not in INCOMPATIBLE_EXTENSIONS:
            continue
        if p.script_args[ext.args_from] is True:
            return False

    return True


class NegPiP(scripts.Script):
    _patched: list[bool] = [False, False]

    def __init__(self):
        self.active: bool = False

        self.is_xl: bool
        self.is_anima: bool
        self.is_hr: bool

        self.tokenizer: torch.nn.Module

        self.has_hr_p: bool
        self.has_hr_n: bool
        self.rev: bool
        self.batch_size: int

        # The current step's negative targets joined into one block (None
        # when it has none) and its token count; see _calc_conds
        self.conds: torch.Tensor | None
        self.c_len: int
        self.c_tokens: int
        self.conds_all: list[list[tuple[int, torch.Tensor | None, int]]]
        self.hr_conds_all: list[list[tuple[int, torch.Tensor | None, int]]]

        self.unconds: torch.Tensor | None
        self.uc_len: int
        self.uc_tokens: int
        self.unconds_all: list[list[tuple[int, torch.Tensor | None, int]]]
        self.hr_unconds_all: list[list[tuple[int, torch.Tensor | None, int]]]

        on_cfg_denoiser(self.denoiser_callback)

    def reset(self):
        self.active = False

        self.is_xl = False
        self.is_anima = False
        self.is_hr = False

        self.tokenizer = None

        self.conds = None
        self.c_tokens = None
        self.conds_all = None
        self.hr_conds_all = None

        self.unconds = None
        self.uc_tokens = None
        self.unconds_all = None
        self.hr_unconds_all = None

        patch_sd_negpip(None, NegPiP, unpatch=True)
        patch_anima_negpip(NegPiP, unpatch=True)

    def title(self):
        return "NegPiP"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        return None

    def process_batch(self, p: "StableDiffusionProcessing", *args, **kwargs):
        self.reset()

        if not any_negative(p):
            return

        if not _verify_ext(p):
            logger.warning("NegPiP Disabled: an incompatible extension is enabled")
            return

        if IS_NEO and not p.sd_model.is_webui_legacy_model():
            self.is_anima = is_anima_engine(p.sd_model)
            if self.is_anima:
                patch_anima_negpip(NegPiP)
                reset_prompt_cache(p)
                p.extra_generation_params["NegPiP"] = True
                self.active = True
            return

        if not any_negative_target(p):
            return  # only a weight around a nested group: the SD path leaves those to the parser

        self.is_xl = p.sd_model.is_sdxl
        self.batch_size = p.batch_size
        self.has_hr_p, self.has_hr_n = hr_dealer(p)
        self.rev = p.sampler_name not in ("DDIM", "PLMS", "UniPC")

        if IS_NEO:
            self.tokenizer = (
                p.sd_model.text_processing_engine_l.tokenize_line
                if self.is_xl
                else p.sd_model.text_processing_engine.tokenize_line
            )
        else:
            self.tokenizer = (
                p.sd_model.conditioner.embedders[0].tokenize_line
                if self.is_xl
                else p.sd_model.cond_stage_model.tokenize_line
            )

        hr_steps: int = getattr(p, "hr_second_pass_steps", 0) or p.steps
        lists = {
            "prompts": (p.prompts, p.steps),
            "negative prompts": (p.negative_prompts, p.steps),
            "Hires prompts": (p.hr_prompts, hr_steps) if self.has_hr_p else None,
            "Hires negative prompts": (p.hr_negative_prompts, hr_steps) if self.has_hr_n else None,
        }
        scheduled = {}
        for name, entry in lists.items():
            if entry is None:
                scheduled[name] = None
                continue
            scheduled[name], agree = self._getScheduledNegPip(*entry)
            if not agree:
                record_conflict_resolution(p, f"NegPiP skipped the batch's {name}: they weight different negative targets")
        nip, pin, hr_nip, hr_pin = scheduled.values()

        schedules = [schedule for schedule in scheduled.values() if schedule is not None]
        if not any(targets for schedule in schedules for steps_out in schedule for _step, targets in steps_out):
            return  # no group can be moved out (prompt editing, a weighted keyword): the parser keeps those weights

        encoded = {}  # one encode per distinct target, shared by every entry and list
        self.conds_all = self._calc_conds(p, nip, encoded)
        self.unconds_all = self._calc_conds(p, pin, encoded)
        if hr_nip is not None:
            self.hr_conds_all = self._calc_conds(p, hr_nip, encoded)
        if hr_pin is not None:
            self.hr_unconds_all = self._calc_conds(p, hr_pin, encoded)

        def calcChunks(a: int, b: int) -> int:
            return a // b if a % b == 0 else a // b + 1

        self.c_len = calcChunks(self.tokenizer(p.prompts[0])[1], 75)
        self.uc_len = calcChunks(self.tokenizer(p.negative_prompts[0])[1], 75)

        patch_sd_negpip(self, NegPiP)
        reset_prompt_cache(p)
        p.extra_generation_params["NegPiP"] = True
        self.active = True

        if self.conds_all[0][0][2] > 0:
            logger.info("NegPiP Enable (Positive: %s)", self.conds_all[0][0][2])
        if self.unconds_all[0][0][2] > 0:
            logger.info("NegPiP Enable (Negative: %s)", self.unconds_all[0][0][2])

    def postprocess(self, *args, **kwargs):
        # Unhook before later scripts' postprocess, which may start nested generations
        self.reset()

    def cleanup(self, *args, **kwargs):
        # Runs even when generation raises, so hooks never outlive a failed run.
        self.reset()

    def before_hr(self, p: "StableDiffusionProcessing", *args, **kwargs):
        self.is_hr = True
        if self.active and self.is_anima and patched_engine() is not p.sd_model:
            self._rehook_anima_for_hires(p)

    def _rehook_anima_for_hires(self, p: "StableDiffusionProcessing"):
        # A Hires checkpoint (or module set) was loaded after process_batch:
        # the hooks sit on the first pass model, and the Hires conditioning
        # was just encoded by the new, unhooked one.
        patch_anima_negpip(NegPiP, unpatch=True)
        model = p.sd_model
        if model.is_webui_legacy_model() or not is_anima_engine(model):
            self.active = False
            record_conflict_resolution(p, "NegPiP off in Hires: the Hires checkpoint is not Anima")
            return
        patch_anima_negpip(NegPiP)
        reset_prompt_cache(p)  # the Hires lookup falls back to the first pass cache
        p.hr_c = p.hr_uc = None
        with devices.autocast():
            p.calculate_hr_conds()

    def denoiser_callback(self, params: CFGDenoiserParams):
        if (not self.active) or self.is_anima:
            return

        # The step Forge picks the main prompt's entry by (the CFG denoiser's call
        # counter, as reconstruct_cond_batch reads it); state.sampling_step lags
        # it by one call, which made "[(dog:-1)::5]" or "[(dog:-1)|bird]" switch early
        step = getattr(getattr(params, "denoiser", None), "step", None)
        step = params.sampling_step if step is None else step

        conds_all = self.hr_conds_all if self.is_hr and self.has_hr_p else self.conds_all
        if conds_all is not None:
            self.conds, self.c_tokens = self._current_entry(conds_all, step)

        unconds_all = self.hr_unconds_all if self.is_hr and self.has_hr_n else self.unconds_all
        if unconds_all is not None:
            self.unconds, self.uc_tokens = self._current_entry(unconds_all, step)

    @staticmethod
    def _current_entry(
        schedule: list[list[tuple[int, torch.Tensor | None, int]]], step: int
    ) -> tuple[torch.Tensor | None, int]:
        """The joined negative targets of the batch's entry at ``step``, picked
        like Forge picks the main prompt's (schedule_index)."""
        entries = schedule[0]
        _end, conds, tokens = entries[schedule_index([end for end, _conds, _tokens in entries], step)]
        return conds, tokens

    @staticmethod
    def _getScheduledNegPip(
        prompts: list[str], steps: list[int]
    ) -> tuple[list[list[tuple[int, list[tuple[str, float]]]]], bool]:
        """The negative targets of each prompt's schedule entries, moved out of
        ``prompts``, and whether the batch rows agree on them. The SD path
        applies one block of targets to the whole batch (the hook repeats it),
        so rows weighting different targets ("a cat" next to "a cat, (dog:-1)")
        are left untouched with no targets instead of losing or borrowing them."""

        originals = list(prompts)
        output = []

        scheduled = get_learned_conditioning_prompt_schedules(prompts, steps)
        for i, batch_schedule in enumerate(scheduled):
            stepout = []
            raw = prompts[i]

            for step, prompt in batch_schedule:
                text_weights: list[tuple[str, float]] = []

                for match, text, weight in negative_weights(prompt):
                    # A group holding prompt editing ("(dog [big:small:5]:-1)")
                    # is not in the raw prompt as a step reads it: it cannot be
                    # moved out, so the parser keeps its weight (as for nested
                    # groups) instead of NegPiP negating it a second time.
                    # Checked against the raw prompt, so a target removed at an
                    # earlier step is still encoded at the later ones.
                    # (a group of only BREAK weights no text, so it never gets here)
                    if match not in raw or text.strip() in ("AND", "ADDCOL", "ADDROW"):
                        continue  # a weighted keyword is no target: the prompt keeps it too
                    prompts[i] = prompts[i].replace(match, "")
                    text_weights.append((text, weight))

                stepout.append((step, text_weights))

            output.append(stepout)

        if any(row != output[0] for row in output[1:]):
            prompts[:] = originals
            return [[(step, []) for step, _targets in row] for row in output], False
        return output, True

    def _cond_dealer(
        self, p: "StableDiffusionProcessing", target: tuple[str, float]
    ) -> tuple[torch.Tensor, int]:
        conds = []

        input = SdConditioning(
            [f"({target[0]}:{-target[1]})"],
            width=p.width,
            height=p.height,
        )

        cond = get_learned_conditioning(p.sd_model, input, p.steps)

        _, token_len = self.tokenizer(target[0])

        conds.append(
            cond[0][0].cond[1 : token_len + 2, :]
            if not self.is_xl
            else cond[0][0].cond["crossattn"][1 : token_len + 2, :]
        )

        conds = torch.cat(conds, 0).unsqueeze(0)
        return conds.repeat(self.batch_size, 1, 1), conds.shape[1]

    def _calc_conds(
        self,
        p: "StableDiffusionProcessing",
        targetlist: list[list[tuple[int, list[tuple[str, float]]]]],
        encoded: dict | None = None,
    ) -> list[list[tuple[int, torch.Tensor | None, int]]]:
        """(end step, conditioning, tokens) per schedule entry of each prompt.

        Every negative target of an entry is encoded on its own, then the
        targets are joined along the token dimension: the attention hook
        appends that block to the context and negates its last ``tokens``
        values, so all the targets apply (None, 0 when the entry has none).
        ``encoded`` caches each distinct target's encoding (a text-encoder pass
        each) across entries; the tensors are only joined or expanded later."""

        encoded = {} if encoded is None else encoded
        outconds = []
        for batch in targetlist:
            stepconds = []
            for step, targets in batch:
                for target in targets:
                    if target not in encoded:
                        encoded[target] = self._cond_dealer(p, target)
                parts = [encoded[target] for target in targets]
                if parts:
                    conds = torch.cat([cond for cond, _ in parts], 1)
                    stepconds.append((step, conds, sum(tokens for _, tokens in parts)))
                else:
                    stepconds.append((step, None, 0))
            outconds.append(stepconds)
        return outconds
