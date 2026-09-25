import pytest

from modules_forge import cuda_malloc

VARS = cuda_malloc._ALLOC_CONF_VARS


@pytest.fixture
def env(monkeypatch):
    for var in VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(cuda_malloc, "cuda_malloc_supported", lambda: True)
    return monkeypatch


def _set(*names):
    import os

    return {name: os.environ[name] for name in VARS if name in os.environ and (not names or name in names)}


def test_both_options_land_in_one_variable(env):
    cuda_malloc.try_cuda_malloc()
    cuda_malloc.try_expandable_segments()

    assert _set() == {"PYTORCH_ALLOC_CONF": "backend:cudaMallocAsync,expandable_segments:True"}


def test_options_join_a_legacy_user_config(env):
    env.setenv("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.6")

    cuda_malloc.try_expandable_segments()

    # PyTorch reads PYTORCH_CUDA_ALLOC_CONF first and ignores the others
    assert _set() == {"PYTORCH_CUDA_ALLOC_CONF": "garbage_collection_threshold:0.6,expandable_segments:True"}


def test_options_join_a_current_user_config(env):
    env.setenv("PYTORCH_ALLOC_CONF", "max_split_size_mb:128")

    cuda_malloc.try_cuda_malloc()

    assert _set() == {"PYTORCH_ALLOC_CONF": "max_split_size_mb:128,backend:cudaMallocAsync"}


def test_unsupported_gpu_keeps_the_environment(env):
    env.setattr(cuda_malloc, "cuda_malloc_supported", lambda: False)

    cuda_malloc.try_cuda_malloc()

    assert _set() == {}
