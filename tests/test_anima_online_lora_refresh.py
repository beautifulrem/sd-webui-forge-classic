"""Forge's settings refresh must not drop the online-LoRA leases mid-run.

processing.py refreshes the model loading parameters around a Hires checkpoint
or module switch, while the stage scheduler / FreeFuse hold online-LoRA leases;
assigning ``dynamic_args.online_lora`` there loaded the hires LoRAs offline.
"""

import ast
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from backend.args import dynamic_args
from modules import anima_online_lora as leases

_SOURCE = Path(__file__).parents[1] / "modules_forge" / "main_entry.py"
_NAMES = {"forge_unet_storage_dtype_options", "anima_quant_options", "_migrate_scaled_fp8_setting", "forge_loading_parameters", "refresh_model_loading_parameters"}


@pytest.fixture(autouse=True)
def _fresh_leases(monkeypatch):
    monkeypatch.setattr(leases, "_leases", set())
    monkeypatch.setattr(leases, "_user_setting", None)
    monkeypatch.setattr(dynamic_args, "online_lora", False)


def _refresh(monkeypatch, storage_option):
    """main_entry's own refresh_model_loading_parameters against stubs (it only imports in the WebUI)."""
    tree = ast.parse(_SOURCE.read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if getattr(n, "name", None) in _NAMES or (isinstance(n, ast.AnnAssign) and getattr(n.target, "id", None) in _NAMES)]
    opts = SimpleNamespace(data={}, forge_unet_storage_dtype=storage_option, forge_additional_modules=[])
    ns = {
        "torch": torch,
        "os": __import__("os"),
        "logger": logging.getLogger("test_online_lora_refresh"),
        "print_json": lambda **kwargs: None,
        "shared": SimpleNamespace(opts=opts, config_filename="config.json"),
        "processing": SimpleNamespace(),
        "anima_online_lora": leases,
        "dynamic_args": dynamic_args,  # the old code assigned it directly
    }
    exec(compile(ast.Module(nodes, []), str(_SOURCE), "exec"), ns)  # noqa: S102 - our own source file
    sd_models = ModuleType("modules.sd_models")
    sd_models.model_data = SimpleNamespace(forge_loading_parameters={})
    sd_models.select_checkpoint = lambda: SimpleNamespace(filename="anima.safetensors")
    monkeypatch.setitem(sys.modules, "modules.sd_models", sd_models)
    return ns["refresh_model_loading_parameters"]


def test_a_mid_run_refresh_keeps_leased_online_lora_on(monkeypatch):
    p = SimpleNamespace()
    leases.acquire_online_lora(p, "scheduler")
    leases.acquire_online_lora(p, "freefuse")

    _refresh(monkeypatch, "Automatic")()  # Hires checkpoint switch: fp16 LoRA off

    assert dynamic_args.online_lora is True
    leases.release_online_lora(p, "scheduler")
    assert dynamic_args.online_lora is True
    leases.release_online_lora(p, "freefuse")
    assert dynamic_args.online_lora is False


def test_the_refreshed_setting_is_what_the_last_release_restores(monkeypatch):
    p = SimpleNamespace()
    leases.acquire_online_lora(p, "scheduler")  # saved user setting: off

    _refresh(monkeypatch, "float8-e4m3fn (fp16 LoRA)")()  # the user switched fp16 LoRA on mid-run

    assert dynamic_args.online_lora is True and leases.user_online_lora() is True
    leases.release_online_lora(p, "scheduler")
    assert dynamic_args.online_lora is True


def test_without_leases_the_refresh_sets_the_switch(monkeypatch):
    refresh = _refresh(monkeypatch, "float8-e4m3fn (fp16 LoRA)")
    refresh()
    assert dynamic_args.online_lora is True

    _refresh(monkeypatch, "Automatic")()
    assert dynamic_args.online_lora is False
