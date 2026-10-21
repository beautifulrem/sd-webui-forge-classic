"""docker/: the image recipe builds this checkout, Gradio-free.

The image cannot be built in the test suite (it needs a Docker daemon, the
network and gigabytes of CUDA wheels), so these tests read the recipe: it
copies the build context instead of cloning another repository, installs
``requirements.txt`` (which has no Gradio) and the launcher's PyTorch build,
builds the SPA with the launcher's pnpm, keeps all user data in ``--data-dir``
and leaves Gradio-era files (``ui-config.json``) out.
"""

from __future__ import annotations

import re
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCKER = ROOT / "docker"
DOCKERFILE = (DOCKER / "Dockerfile").read_text(encoding="utf-8")
ENTRYPOINT = (DOCKER / "entrypoint.sh").read_text(encoding="utf-8")
README = (DOCKER / "README.md").read_text(encoding="utf-8")
IGNORE = [line.strip() for line in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]


def instructions() -> list[tuple[str, str]]:
    """``(INSTRUCTION, arguments)`` of the Dockerfile, continuation lines joined."""

    joined = re.sub(r"\\\n", " ", DOCKERFILE)
    found = []
    for line in joined.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            keyword, _, rest = line.partition(" ")
            found.append((keyword.upper(), rest.strip()))
    return found


def arg_default(name: str) -> str:
    for keyword, rest in instructions():
        if keyword == "ARG" and rest.startswith(f"{name}="):
            return rest.split("=", 1)[1].strip('"')
    raise AssertionError(f"no ARG {name}")


def test_the_image_is_built_from_this_checkout():
    assert "git clone" not in DOCKERFILE and "github.com" not in DOCKERFILE
    copies = [rest for keyword, rest in instructions() if keyword == "COPY"]
    # the whole build context, and the SPA from the image's own Node.js stage
    assert "--chown=99:100 . ./" in copies
    assert "--from=webui --chown=99:100 /src/webui/dist ./webui/dist" in copies
    # every path copied from the context exists in the checkout
    for rest in copies:
        if "--from=" in rest:
            continue
        sources = [part for part in shlex.split(rest) if not part.startswith("--")][:-1]
        for source in sources:
            assert (ROOT / source).exists(), source


def test_the_python_side_is_the_webuis_requirements():
    runs = " ".join(rest for keyword, rest in instructions() if keyword == "RUN")
    assert "uv pip install -r requirements.txt" in runs
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    assert "gradio" not in requirements
    assert not any("gradio" in rest.lower() for _keyword, rest in instructions())
    # the built-in extensions' installers run through the launcher, which then exits
    assert "python launch.py --exit --skip-torch-cuda-test --data-dir /tmp/remi-build" in runs


def test_pytorch_and_pnpm_follow_the_launcher():
    from modules import launch_utils

    command = launch_utils.default_torch_command(platform="linux")
    packages = arg_default("TORCH_PACKAGES")
    assert f"pip install {packages} --extra-index-url {arg_default('TORCH_INDEX_URL')}" == command
    assert f"pnpm@{arg_default('PNPM_VERSION')}" == launch_utils.PNPM_FALLBACK
    assert launch_utils.node_version_problem(f"v{arg_default('NODE_VERSION')}.0.0") is None
    # the image ships the build: the launcher must not try to rebuild it without Node.js
    assert "REMI_SKIP_WEBUI_BUILD=1" in DOCKERFILE
    assert "pnpm install --frozen-lockfile" in DOCKERFILE and "pnpm run build" in DOCKERFILE
    # the base image's CUDA is the wheels' CUDA
    major, minor = arg_default("CUDA_VERSION").split(".")[:2]
    assert arg_default("TORCH_INDEX_URL").endswith(f"/cu{major}{minor}")


def test_all_user_data_is_in_the_data_folder():
    assert "ui-config.json" not in ENTRYPOINT and "ln -s" not in ENTRYPOINT
    assert '--data-dir "$REMI_DATA_DIR"' in ENTRYPOINT and "--listen" in ENTRYPOINT
    assert "REMI_DATA_DIR=/home/remi/data" in DOCKERFILE and 'VOLUME ["/home/remi/data"]' in DOCKERFILE
    # the healthcheck uses the one route that answers without a login
    from remi.server.auth import PUBLIC_PATHS

    assert "/internal/ping" in PUBLIC_PATHS and "localhost:7860/internal/ping" in DOCKERFILE


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_the_entrypoint_parses():
    assert subprocess.run(["bash", "-n", str(DOCKER / "entrypoint.sh")], capture_output=True).returncode == 0


def test_the_build_context_leaves_out_data_and_build_output():
    for pattern in ("/models", "/extensions", "/output", "/remi-data", "/config.json", "/venv", "/.venv", "/webui/node_modules", "/webui/dist", ".git", "**/*.safetensors"):
        assert pattern in IGNORE, pattern


def test_the_readme_describes_this_image():
    assert "Forge - Neo" not in README and "git clone" not in README
    assert "docker build -f docker/Dockerfile -t remi-webui ." in README
    assert "--data-dir" in README
