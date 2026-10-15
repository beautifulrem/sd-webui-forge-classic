import os
import sys

# The WebUI parses sys.argv when modules.shared is first imported; keep
# pytest's own arguments away from it.
sys.argv = sys.argv[:1]
# As modules_forge/initialization.py does at startup (huggingface_guess, k_diffusion, ...)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "modules_forge", "packages"))

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_api_auth_throttle():
    """Failed --api-auth logins are counted process-wide; tests start clean."""
    from remi.server.auth import API_AUTH_THROTTLE

    API_AUTH_THROTTLE._failures.clear()
    yield
