"""
install.py for sd-webui-joycaption

This is auto-run by Forge / A1111 on launch (once). It ensures the
Python packages the extension needs are present. Forge already ships
with `transformers`, `accelerate`, `torch`, `huggingface-hub`, and
`pillow`, so usually this is a no-op. `bitsandbytes` is optional and
only needed for 4-bit / 8-bit quantization (saves VRAM); if absent,
those options are disabled in the UI.
"""

import importlib.util
import sys

try:
    import launch  # provided by webui at runtime
except ImportError:
    launch = None


def _have(pkg: str) -> bool:
    return importlib.util.find_spec(pkg) is not None


def _ensure(pkg_import_name: str, pip_name: str = None, version_spec: str = ""):
    pip_name = pip_name or pkg_import_name
    if _have(pkg_import_name):
        return
    if launch is None:
        # Fallback if run outside webui
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", f"{pip_name}{version_spec}"])
    else:
        launch.run_pip(f"install {pip_name}{version_spec}", f"sd-webui-joycaption: {pip_name}")


# Required (almost certainly already installed by Forge):
_ensure("transformers", version_spec=">=4.45")
_ensure("accelerate")
_ensure("huggingface_hub", "huggingface-hub")
_ensure("PIL", "Pillow")

# bitsandbytes is optional. Forge Neo already owns its installation through
# the explicit --bnb launch option; do not silently mutate the environment on
# every extension install. The UI detects its absence and hides int8/nf4.
