import os
import numpy as np
from tqdm import tqdm

import modules.scripts as scripts
from modules.script_callbacks import on_cfg_denoiser, remove_callbacks_for_function, on_infotext_pasted, on_ui_settings
from modules import shared
from remi import ui

try:
    import modules_forge.forge_version  # noqa: F401 - only probes for Forge
    is_forge = True
except:
    is_forge = False

def add_settings():
    section = ('detail_daemon', "Detail Daemon")
    shared.opts.add_option("detail_daemon_count", shared.OptionInfo(
        6, "Daemon count", ui.Slider, {"minimum": 1, "maximum": 10, "step": 1}, section=section).needs_reload_ui())
    shared.opts.add_option("detail_daemon_verbose", shared.OptionInfo(
        False, "Verbose", ui.Toggle, section=section))

on_ui_settings(add_settings)

def parse_infotext(infotext, params):
    try:
        raw = params['Detail Daemon']
        if raw.startswith("D"):
            daemons = raw.split(";")
            if len(daemons) > shared.opts.data.get("detail_daemon_count", 6):
                tqdm.write(f"\033[31mDetail Daemon:\033[0m Need more daemons! Go to 'Settings > Uncategorized > Detail Daemon' and increase count to {len(daemons)}.")
            dd_dict = {}
            for idx, daemon in enumerate(daemons):
                tag, values = daemon.split(":")
                vals = values.split(",")
                dd_dict[f"active{idx + 1 if idx > 0 else ''}"] = bool(int(vals[0]))
                dd_dict[f"hr{idx + 1 if idx > 0 else ''}"] = bool(int(vals[1]))
                dd_dict[f"mode{idx + 1 if idx > 0 else ''}"] = vals[2]
                dd_dict[f"amount{idx + 1 if idx > 0 else ''}"] = float(vals[3])
                dd_dict[f"st{idx + 1 if idx > 0 else ''}"] = float(vals[4])
                dd_dict[f"ed{idx + 1 if idx > 0 else ''}"] = float(vals[5])
                dd_dict[f"bias{idx + 1 if idx > 0 else ''}"] = float(vals[6])
                dd_dict[f"exp{idx + 1 if idx > 0 else ''}"] = float(vals[7])
                dd_dict[f"st_offset{idx + 1 if idx > 0 else ''}"] = float(vals[8])
                dd_dict[f"ed_offset{idx + 1 if idx > 0 else ''}"] = float(vals[9])
                dd_dict[f"fade{idx + 1 if idx > 0 else ''}"] = float(vals[10])
                dd_dict[f"smooth{idx + 1 if idx > 0 else ''}"] = bool(int(vals[11]))
            params['Detail Daemon'] = dd_dict
        else:
            # fallback to old format for backward compatibility
            d = {}
            for s in raw.split(','):
                k, _, v = s.partition(':')
                d[k.strip()] = v.strip()
            params['Detail Daemon'] = d
    except Exception:
        pass

on_infotext_pasted(parse_infotext)


ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]

DAEMON_PARAMS = ("active", "hires", "mode", "start", "end", "bias", "amount", "exponent", "start_offset", "end_offset", "fade", "smooth")
"""One daemon's arguments, in process() order (param names are ``d<n>_<name>``)."""

INFOTEXT_KEYS = {
    "active": "active", "hires": "hr", "mode": "mode", "start": "st", "end": "ed", "bias": "bias", "amount": "amount",
    "exponent": "exp", "start_offset": "st_offset", "end_offset": "ed_offset", "fade": "fade", "smooth": "smooth",
}
"""Param → key of the parsed ``Detail Daemon`` infotext (daemon 2 and up append their number)."""

LEGACY_KEYS = {
    "mode": "DD_mode", "amount": "DD_amount", "start": "DD_start", "end": "DD_end", "bias": "DD_bias", "exponent": "DD_exponent",
    "start_offset": "DD_start_offset", "end_offset": "DD_end_offset", "fade": "DD_fade", "smooth": "DD_smooth",
}
"""The first daemon also reads the infotext of the single-daemon versions (``DD_*`` keys)."""


def daemon_count() -> int:
    return min(max(int(shared.opts.data.get("detail_daemon_count", 6)), 1), len(ROMAN))


def extract_infotext(d: dict, key, old_key=None):
    parsed = d.get('Detail Daemon')
    if parsed is not None:
        return parsed.get(key) if isinstance(parsed, dict) else None
    return d.get(old_key)


def _paste(index: int, name: str):
    """The paste function of daemon ``index``'s param ``name``, as the Gradio UI read it."""

    if index == 0:
        key, old_key = INFOTEXT_KEYS[name], LEGACY_KEYS.get(name)
        if name == "active":
            return lambda d: extract_infotext(d, key) or 'Detail Daemon' in d or 'DD_enabled' in d
        if name == "hires":
            return lambda d: extract_infotext(d, key) or False
        return lambda d: extract_infotext(d, key, old_key)
    key = f"{INFOTEXT_KEYS[name]}{index + 1}"
    if name == "active":
        return lambda d: extract_infotext(d, key) or False
    return lambda d: extract_infotext(d, key)


def daemon_params(index: int):
    """The controls of daemon ``index`` (a tab)."""

    def name(param):
        return f"d{index + 1}_{param}"

    def common(param):
        return {"paste": _paste(index, param)}

    return [
        ui.Row(
            ui.Toggle(name("active"), False, label="Active", **common("active")),
            ui.Toggle(name("hires"), False, label="Hires Pass", **common("hires")),
            ui.Toggle(name("smooth"), True, label="Smooth", **common("smooth")),
            ui.Choice(name("mode"), ["both", "cond", "uncond"], "both", label="Mode", **common("mode")),
        ),
        # Numbers, not sliders: the amount and offsets may go past -1..1
        ui.Row(
            ui.Number(name("amount"), 0.10, step=.01, precision=4, label="Amount", **common("amount")),
            ui.Number(name("start_offset"), 0.0, step=.01, precision=4, label="Start Offset", **common("start_offset")),
            ui.Number(name("end_offset"), 0.0, step=.01, precision=4, label="End Offset", **common("end_offset")),
        ),
        ui.Row(
            ui.Slider(name("start"), 0.0, 1.0, 0.2, .01, label="Start", **common("start")),
            ui.Slider(name("end"), 0.0, 1.0, 0.8, .01, label="End", **common("end")),
        ),
        ui.Row(
            ui.Slider(name("bias"), 0.0, 1.0, 0.5, .01, label="Bias", **common("bias")),
            ui.Slider(name("exponent"), 0.0, 10.0, 1.0, .05, label="Exponent", **common("exponent")),
        ),
        ui.Slider(name("fade"), 0.0, 1.0, 0.0, .05, label="Fade", **common("fade")),
    ]


class Script(scripts.Script):

    def __init__(self):
        self.tab_param_count = len(DAEMON_PARAMS)
        self.param_order = None

    def title(self):
        return "Detail Daemon"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def params(self, is_img2img):
        count = daemon_count()
        self.tab_param_count = len(DAEMON_PARAMS)
        # process() reads the enable toggle, then each daemon's params; the
        # curve preview and the selected tab come last (never read).
        self.param_order = ("enabled", *(f"d{i + 1}_{param}" for i in range(count) for param in DAEMON_PARAMS), "daemon", "curves")
        return ui.Section(
            "Detail Daemon",
            enable=ui.Toggle("enabled", False, label="Detail Daemon", paste=lambda d: 'Detail Daemon' in d or 'DD_enabled' in d),
            children=[
                ui.Custom(
                    "curves",
                    "sigma-curve",
                    None,
                    props={"daemons": [{"prefix": f"d{i + 1}_", "title": ROMAN[i], "value": str(i + 1)} for i in range(count)], "tab": "daemon", "enable": "enabled"},
                    label="",
                ),
                ui.Choice("daemon", [(ROMAN[i], str(i + 1)) for i in range(count)], "1", label="Daemon", hidden=True),
                ui.Tabs(*(ui.Tab(ROMAN[i], daemon_params(i), value=str(i + 1)) for i in range(count)), bind="daemon"),
            ],
        )

    def process(self, p, enabled, *all_daemon_args):
        if not enabled:
            self._remove_callback()
            return

        if p.sampler_name in ["DPM adaptive", "HeunPP2"]:
            tqdm.write(f'\033[31mDetail Daemon:\033[0m Selected sampler ({p.sampler_name}) is not supported.')
            self._remove_callback()
            return

        self.daemon_data = []
        extra_gen_texts = []
        num_daemons = len(all_daemon_args) // self.tab_param_count

        for i in range(num_daemons):
            start_idx = i * self.tab_param_count
            end_idx = start_idx + self.tab_param_count
            daemon_args = all_daemon_args[start_idx:end_idx]

            active, hires, mode, start, end, bias, amount, exponent, start_offset, end_offset, fade, smooth = daemon_args

            # TODO? XYZ support for other channels
            if (i == 0) :
                mode = getattr(p, "DD_mode", mode)
                amount = getattr(p, "DD_amount", amount)
                start = getattr(p, "DD_start", start)
                end = getattr(p, "DD_end", end)
                bias = getattr(p, "DD_bias", bias)
                exponent = getattr(p, "DD_exponent", exponent)
                start_offset = getattr(p, "DD_start_offset", start_offset)
                end_offset = getattr(p, "DD_end_offset", end_offset)
                fade = getattr(p, "DD_fade", fade)
                smooth = getattr(p, "DD_smooth", smooth)

            if active:
                daemon_schedule_params = {
                    "start": start,
                    "end": end,
                    "bias": bias,
                    "amount": amount,
                    "exponent": exponent,
                    "start_offset": start_offset,
                    "end_offset": end_offset,
                    "fade": fade,
                    "smooth": smooth
                }

                self.daemon_data.append({
                    'name': f'Daemon {i+1}',
                    'mode': mode,
                    'schedule': None,
                    'schedule_params': daemon_schedule_params,
                    'hires': hires,
                    'multiplier': .1  # Add slider for this?
                })

                text = ",".join([
                    str(int(active)), str(int(hires)), mode, f"{amount}", f"{start}", f"{end}", f"{bias}",
                    f"{exponent}", f"{start_offset}", f"{end_offset}", f"{fade:}", str(int(smooth))
                ])
                extra_gen_texts.append(f"D{i+1}:{text}")

        if extra_gen_texts:
            p.extra_generation_params['Detail Daemon'] = ";".join(extra_gen_texts)

        if not hasattr(self, 'callback_added'):
            on_cfg_denoiser(self.denoiser_callback)
            self.callback_added = True
        self.cfg_scale = p.cfg_scale
        self.batch_size = p.batch_size
        self.is_hires_pass = False

    def before_process_batch(self, p, *args, **kwargs):
        self.is_hires_pass = False

    def before_hr(self, p, *args):
        self.is_hires_pass = True

    def _remove_callback(self):
        if hasattr(self, 'callback_added'):
            remove_callbacks_for_function(self.denoiser_callback)
            delattr(self, 'callback_added')

    def postprocess(self, p, processed, *args):
        self._remove_callback()

    def cleanup(self, p, *args):
        # Runs even when generation raises, so the global callback never leaks.
        self._remove_callback()

    def denoiser_callback(self, params):
        for daemon in self.daemon_data:
            if daemon['hires'] != self.is_hires_pass:
                continue

            name = daemon['name']
            mode = daemon['mode']
            step = max(params.sampling_step, params.denoiser.step)
            steps = max(params.total_sampling_steps, params.denoiser.total_steps)
            actual_steps = steps - max(steps // params.denoiser.steps - 1, 0)
            idx = min(step, actual_steps - 1)

            if daemon['schedule'] is None:
                daemon['schedule'] = self.make_schedule(actual_steps, **daemon['schedule_params'])

            schedule = daemon['schedule']
            multiplier = schedule[idx] * daemon['multiplier']

            if is_forge:
                if idx == 0 and mode != "both":
                    tqdm.write('\033[33mDetail Daemon:\033[0m Forge does not support `cond` and `uncond` modes, using `both` instead')
                mode = "both"

            if mode == "cond":
                for i in range(self.batch_size):
                    params.sigma[i] *= 1 - multiplier
            elif mode == "uncond":
                for i in range(self.batch_size):
                    params.sigma[self.batch_size + i] *= 1 + multiplier
            else:
                params.sigma *= 1 - multiplier * self.cfg_scale

            if shared.opts.data.get("detail_daemon_verbose", False):
                tqdm.write(f'\033[32mDetail Daemon:\033[0m {name} | sigma: {params.sigma} | step: {idx}/{actual_steps - 1} | multiplier: {multiplier:.4f}')

    def make_schedule(self, steps, start, end, bias, amount, exponent, start_offset, end_offset, fade, smooth):
        start = min(start, end)
        mid = start + bias * (end - start)
        multipliers = np.zeros(steps)

        start_idx, mid_idx, end_idx = [int(round(x * (steps - 1))) for x in [start, mid, end]]

        start_values = np.linspace(0, 1, mid_idx - start_idx + 1)
        if smooth:
            start_values = 0.5 * (1 - np.cos(start_values * np.pi))
        start_values = start_values ** exponent
        if start_values.any():
            start_values *= (amount - start_offset)
            start_values += start_offset

        end_values = np.linspace(1, 0, end_idx - mid_idx + 1)
        if smooth:
            end_values = 0.5 * (1 - np.cos(end_values * np.pi))
        end_values = end_values ** exponent
        if end_values.any():
            end_values *= (amount - end_offset)
            end_values += end_offset

        multipliers[start_idx:mid_idx+1] = start_values
        multipliers[mid_idx:end_idx+1] = end_values
        multipliers[:start_idx] = start_offset
        multipliers[end_idx+1:] = end_offset
        multipliers *= 1 - fade

        return multipliers


def xyz_support():
    for scriptDataTuple in scripts.scripts_data:
        if os.path.basename(scriptDataTuple.path) == 'xyz_grid.py':
            xy_grid = scriptDataTuple.module

            def confirm_mode(p, xs):
                for x in xs:
                    if x not in ['both', 'cond', 'uncond']:
                        raise RuntimeError(f'Invalid Detail Daemon Mode: {x}')
            mode = xy_grid.AxisOption(
                '[Detail Daemon] Mode',
                str,
                xy_grid.apply_field('DD_mode'),
                confirm=confirm_mode
            )
            amount = xy_grid.AxisOption(
                '[Detail Daemon] Amount',
                float,
                xy_grid.apply_field('DD_amount')
            )
            start = xy_grid.AxisOption(
                '[Detail Daemon] Start',
                float,
                xy_grid.apply_field('DD_start')
            )
            end = xy_grid.AxisOption(
                '[Detail Daemon] End',
                float,
                xy_grid.apply_field('DD_end')
            )
            bias = xy_grid.AxisOption(
                '[Detail Daemon] Bias',
                float,
                xy_grid.apply_field('DD_bias')
            )
            exponent = xy_grid.AxisOption(
                '[Detail Daemon] Exponent',
                float,
                xy_grid.apply_field('DD_exponent')
            )
            start_offset = xy_grid.AxisOption(
                '[Detail Daemon] Start Offset',
                float,
                xy_grid.apply_field('DD_start_offset')
            )
            end_offset = xy_grid.AxisOption(
                '[Detail Daemon] End Offset',
                float,
                xy_grid.apply_field('DD_end_offset')
            )
            fade = xy_grid.AxisOption(
                '[Detail Daemon] Fade',
                float,
                xy_grid.apply_field('DD_fade')
            )
            smooth = xy_grid.AxisOption(
                '[Detail Daemon] Smooth',
                bool,
                xy_grid.apply_field('DD_smooth')
            )
            xy_grid.axis_options.extend([
                mode,
                amount,
                start,
                end,
                bias,
                exponent,
                start_offset,
                end_offset,
                fade,
                smooth,
            ])


try:
    xyz_support()
except Exception as e:
    tqdm.write(f'f"\033[31mDetail Daemon:\033[0m Error trying to add XYZ plot options for Detail Daemon: {e}')
