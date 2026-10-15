"""NegPiP has no panel: no parameters, no legacy ui(), no Gradio."""

from pathlib import Path

import remi_port_harness as harness

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "negpip.py"


def test_the_runner_gives_it_an_empty_ported_layout(monkeypatch):
    from modules import scripts
    from remi.testing import webui_state

    negpip = harness.load_script(SCRIPT, "negpip_params_under_test")
    monkeypatch.setattr(negpip, "on_cfg_denoiser", lambda callback: None)  # keep the session's callbacks
    assert not scripts.has_params(negpip.NegPiP) and not scripts.legacy_ui_only(negpip.NegPiP())
    with webui_state(script_classes=[negpip.NegPiP]) as webui:
        script = next(script for script in webui.txt2img.scripts if isinstance(script, negpip.NegPiP))
        assert script.layout is not None and len(script.layout) == 0


def test_imports_standalone():
    harness.assert_imports_standalone([SCRIPT])
