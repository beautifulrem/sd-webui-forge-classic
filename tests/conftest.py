import os
import sys

# The WebUI parses sys.argv when modules.shared is first imported; keep
# pytest's own arguments away from it.
sys.argv = sys.argv[:1]
# As the WebUI does at startup (modules/launch_utils.py): no analytics or online
# version check from Gradio, which otherwise holds the process open at exit
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
# As modules_forge/initialization.py does at startup (huggingface_guess, k_diffusion, ...)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "modules_forge", "packages"))
