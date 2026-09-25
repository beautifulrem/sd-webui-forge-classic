# reference: https://github.com/Comfy-Org/ComfyUI/blob/master/cuda_malloc.py

import importlib.util
import os.path
import subprocess


def get_gpu_names() -> set[str]:
    gpu_names = set()
    out = subprocess.check_output(["nvidia-smi", "-L"])
    for l in out.split(b"\n"):
        if len(l) > 0:
            gpu_names.add(l.decode("utf-8").split(" (UUID")[0])
    return gpu_names


def cuda_malloc_supported() -> bool:
    blacklist = {"GeForce GTX TITAN X", "GeForce GTX 980", "GeForce GTX 970", "GeForce GTX 960", "GeForce GTX 950", "GeForce 945M", "GeForce 940M", "GeForce 930M", "GeForce 920M", "GeForce 910M", "GeForce GTX 750", "GeForce GTX 745", "Quadro K620", "Quadro K1200", "Quadro K2200", "Quadro M500", "Quadro M520", "Quadro M600", "Quadro M620", "Quadro M1000", "Quadro M1200", "Quadro M2000", "Quadro M2200", "Quadro M3000", "Quadro M4000", "Quadro M5000", "Quadro M5500", "Quadro M6000", "GeForce MX110", "GeForce MX130", "GeForce 830M", "GeForce 840M", "GeForce GTX 850M", "GeForce GTX 860M", "GeForce GTX 1650", "GeForce GTX 1630", "Tesla M4", "Tesla M6", "Tesla M10", "Tesla M40", "Tesla M60"}

    try:
        names = get_gpu_names()
    except Exception:
        names = set()
    for x in names:
        if "NVIDIA" in x:
            for b in blacklist:
                if b in x:
                    return False
    return True


try:
    torch_spec = importlib.util.find_spec("torch")
    for folder in torch_spec.submodule_search_locations:
        ver_file = os.path.join(folder, "version.py")
        if os.path.isfile(ver_file):
            spec = importlib.util.spec_from_file_location("torch_version_import", ver_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            version = module.__version__
except Exception:
    version = ""


# PyTorch parses only the first of these that is set (c10/core/AllocatorConfig.cpp);
# the legacy names come first, PYTORCH_ALLOC_CONF is the current one
_ALLOC_CONF_VARS = ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF", "PYTORCH_ALLOC_CONF")


def _append_alloc_conf(option: str):
    """Add ``option`` to the allocator config PyTorch will actually read, so it
    neither hides nor is hidden by a config the user set under another name"""
    name = next((var for var in _ALLOC_CONF_VARS if var in os.environ), "PYTORCH_ALLOC_CONF")
    env_var = os.environ.get(name)
    os.environ[name] = option if not env_var else f"{env_var},{option}"


def try_cuda_malloc():
    if not cuda_malloc_supported():
        return

    _append_alloc_conf("backend:cudaMallocAsync")


def try_expandable_segments():
    _append_alloc_conf("expandable_segments:True")


def get_torch_version() -> str:
    return str(version)
