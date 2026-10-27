"""Exercise launcher selection without installing packages or starting the UI."""

from __future__ import annotations

import ast
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load_launch_helpers():
    source = ast.parse((ROOT / "modules/launch_utils.py").read_text())
    nodes = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in {"default_torch_command", "_torch_version", "prepare_environment"}]
    namespace = {"os": os, "sys": sys, "re": __import__("re")}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "launch_utils.py", "exec"), namespace)
    return namespace


class TorchInstallTests(unittest.TestCase):
    def setUp(self):
        self.helpers = load_launch_helpers()

    def test_macos_uses_native_wheels_even_with_cuda_index_configured(self):
        with patch.dict(os.environ, {"TORCH_INDEX_URL": "https://example.invalid/cu130"}):
            command = self.helpers["default_torch_command"](platform="darwin")
            self.assertEqual(command, "pip install torch torchvision")

    def test_linux_and_windows_keep_upstream_cuda_versions(self):
        with patch.dict(os.environ, {"TORCH_INDEX_URL": "https://example.invalid/cu130"}):
            for platform in ("linux", "win32"):
                command = self.helpers["default_torch_command"](platform=platform)
                self.assertIn("torch==2.14.0+cu130", command)
                self.assertIn("torchvision==0.29.0+cu130", command)
                self.assertIn("--extra-index-url https://example.invalid/cu130", command)

    def test_nunchaku_uses_the_same_cuda_pair(self):
        # upstream a02a0516 dropped the older PyTorch pair for --nunchaku (Nunchaku 1.4.0 wheels follow torch)
        self.assertEqual(self.helpers["default_torch_command"](platform="linux").count("torch=="), 1)
        self.assertNotIn("2.11.0", self.helpers["default_torch_command"](platform="linux"))

    def test_native_and_accelerator_version_strings(self):
        versions = {
            "2.13.0": ("2.13.0", "cpu"),
            "2.13.0+cu130": ("2.13.0", "cu130"),
            "2.10.0.dev20251111+cu130": ("2.10.0", "cu130"),
            "2.13.0.dev20260930": ("2.13.0", "cpu"),
            "2.13.0+rocm7.0": ("2.13.0", "rocm7.0"),
        }
        for version, expected in versions.items():
            with self.subTest(version=version), patch("importlib.metadata.version", return_value=version):
                self.assertEqual(self.helpers["_torch_version"](), expected)

    def test_custom_torch_command_reaches_installer_unchanged(self):
        class InstallationCaptured(Exception):
            pass

        commands = []

        def capture(command, *args, **kwargs):
            commands.append(command)
            raise InstallationCaptured

        with tempfile.TemporaryDirectory() as directory:
            self.helpers.update(
                script_path=directory,
                startup_timer=SimpleNamespace(record=lambda label: None),
                git_tag=lambda: "test",
                python="/custom/python",
                run=capture,
            )
            for nunchaku in (False, True):
                self.helpers["args"] = SimpleNamespace(nunchaku=nunchaku, skip_python_version_check=True, reinstall_torch=True)
                with self.subTest(nunchaku=nunchaku), patch.dict(os.environ, {"TORCH_COMMAND": "pip install custom-torch"}):
                    with self.assertRaises(InstallationCaptured):
                        self.helpers["prepare_environment"]()
                self.assertEqual(commands[-1], '"/custom/python" -m pip install custom-torch')


@unittest.skipUnless(os.name == "posix", "Unix launcher")
class UnixLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="remi launch ")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.repo = self.base / "repository with spaces"
        self.bin = self.base / "bin"
        self.repo.mkdir()
        self.bin.mkdir()
        for name in ("webui.sh", "webui-user.sh"):
            shutil.copy2(ROOT / name, self.repo / name)
        self.log = self.base / "calls.txt"
        self.env = dict(os.environ)
        for name in ("PYTHON", "VENV_DIR", "SKIP_VENV", "COMMANDLINE_ARGS", "SD_WEBUI_RESTARTING"):
            self.env.pop(name, None)
        self.env.update(PATH=f"{self.bin}:/usr/bin:/bin", TASK_LAUNCH_LOG=str(self.log))
        self.python_stub = self.base / "python stub"
        self.write_executable(self.python_stub, '''#!/bin/bash
echo "$0|$PWD|$*|${COMMANDLINE_ARGS:-}" >> "$TASK_LAUNCH_LOG"
if [ "$1" = "-c" ] && [[ "$2" == *sys.executable* ]]; then
    echo "$0"
fi
if [ "$1" = "-m" ] && [ "$2" = "venv" ]; then
    mkdir -p "$3/bin"
    cp "$0" "$3/bin/python"
    echo ':' > "$3/bin/activate"
fi
''')
        self.env["TASK_TEST_PYTHON_STUB"] = str(self.python_stub)
        self.write_executable(self.bin / "python3", "#!/bin/bash\nexit 1\n")

    def write_executable(self, path, source):
        path.write_text(source)
        path.chmod(0o755)

    def add_python313(self):
        shutil.copy2(self.python_stub, self.bin / "python3.13")

    def add_uv(self, found_python=""):
        self.env["TASK_FOUND_PYTHON"] = str(found_python)
        self.write_executable(self.bin / "uv", '''#!/bin/bash
echo "uv|$PWD|$*" >> "$TASK_LAUNCH_LOG"
if [ "$1" = "python" ] && [ "$2" = "find" ]; then
    [ -n "$TASK_FOUND_PYTHON" ] || exit 1
    echo "$TASK_FOUND_PYTHON"
elif [ "$1" = "venv" ]; then
    mkdir -p "$2/bin"
    cp "$TASK_TEST_PYTHON_STUB" "$2/bin/python"
    echo ':' > "$2/bin/activate"
else
    exit 1
fi
''')

    def run_launcher(self, name="webui.sh", *args):
        return subprocess.run(["/bin/bash", str(self.repo / name), *args], cwd=self.base, env=self.env, capture_output=True, text=True, timeout=10)

    def calls(self):
        return self.log.read_text() if self.log.exists() else ""

    def assert_launched(self, result, python):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(f"{python}|{self.repo}|launch.py", self.calls())

    def test_prefers_python313_over_old_system_python(self):
        self.add_python313()
        result = self.run_launcher("webui.sh", "--api")
        self.assert_launched(result, self.repo / "venv/bin/python")
        self.assertIn(f"{self.bin / 'python3.13'}|{self.repo}|-m venv", self.calls())
        self.assertIn("launch.py --api", self.calls())

    def test_reuses_existing_venv_without_creating_one(self):
        venv = self.repo / "venv/bin"
        venv.mkdir(parents=True)
        shutil.copy2(self.python_stub, venv / "python")
        (venv / "activate").write_text(":\n")
        self.assert_launched(self.run_launcher(), venv / "python")
        self.assertNotIn("-m venv", self.calls())

    def test_explicit_interpreter_and_venv_path_with_spaces(self):
        self.add_python313()
        self.env.update(PYTHON=str(self.python_stub), VENV_DIR=str(self.base / "custom venv"))
        self.assert_launched(self.run_launcher(), self.base / "custom venv/bin/python")
        self.assertIn(f"{self.python_stub}|{self.repo}|-m venv", self.calls())

    def test_uv_finds_an_installed_python313(self):
        self.add_uv(self.python_stub)
        self.assert_launched(self.run_launcher(), self.repo / "venv/bin/python")
        self.assertIn("python find --offline 3.13", self.calls())
        self.assertNotIn("uv|" + str(self.repo) + "|venv", self.calls())

    def test_uv_creates_seeded_python313_environment(self):
        self.add_uv()
        self.assert_launched(self.run_launcher(), self.repo / "venv/bin/python")
        self.assertIn(f"venv {self.repo / 'venv'} --python 3.13 --seed", self.calls())

    def test_skip_venv_uses_selected_python(self):
        self.add_python313()
        self.env["SKIP_VENV"] = "1"
        self.assert_launched(self.run_launcher(), self.bin / "python3.13")
        self.assertNotIn("-m venv", self.calls())

    def test_disabled_venv_does_not_download_python(self):
        self.add_uv()
        self.env["VENV_DIR"] = "-"
        result = self.run_launcher()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Python 3.13 was not found", result.stdout)
        self.assertNotIn("|venv", self.calls())
        self.assertNotIn("launch.py", self.calls())

    def test_missing_python313_fails_before_creating_venv(self):
        result = self.run_launcher()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Python 3.13 was not found", result.stdout)
        self.assertFalse((self.repo / "venv").exists())

    def test_settings_are_loaded_from_repo_when_called_elsewhere(self):
        (self.repo / "webui.settings.sh").write_text(f'export PYTHON="{self.python_stub}"\nexport SKIP_VENV=1\nexport COMMANDLINE_ARGS="--api"\n')
        self.assert_launched(self.run_launcher(), self.python_stub)
        self.assertIn("|launch.py|--api", self.calls())

    def test_user_wrapper_keeps_arguments_and_custom_flags(self):
        self.add_python313()
        self.env.update(SKIP_VENV="1", COMMANDLINE_ARGS="--api")
        self.assert_launched(self.run_launcher("webui-user.sh", "--port", "7870"), self.bin / "python3.13")
        self.assertIn("|launch.py --port 7870|--api", self.calls())


class VerifyVersionTests(unittest.TestCase):
    """``verify_version`` writes config.json first thing at startup: a new
    ``--data-dir`` did not exist yet and launch.py died with FileNotFoundError."""

    def load(self, settings_file):
        source = ast.parse((ROOT / "modules/launch_utils.py").read_text())
        nodes = [node for node in source.body if (isinstance(node, ast.FunctionDef) and node.name == "verify_version") or (isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == "VERSION_UID")]
        namespace = {"os": os, "json": __import__("json"), "Any": object, "Final": object, "args": SimpleNamespace(ui_settings_file=str(settings_file))}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "launch_utils.py", "exec"), namespace)
        return namespace

    def test_a_new_data_dir_is_created(self):
        import json

        with tempfile.TemporaryDirectory() as root:
            settings = Path(root, "new", "data-dir", "config.json")
            helpers = self.load(settings)
            helpers["verify_version"]()
            self.assertEqual(json.loads(settings.read_text()), {"VERSION_UID": helpers["VERSION_UID"]})
            # the next start finds it up to date
            helpers["verify_version"]()


if __name__ == "__main__":
    unittest.main()
