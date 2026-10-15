"""pytest's arguments kept away from the WebUI's argv parser."""

import sys

sys.argv = sys.argv[:1]
