from pathlib import Path

from modules import scripts, shared
from modules.paths import data_path, extensions_builtin_dir, extensions_dir, script_path

# Webui root path
FILE_DIR = Path(script_path).absolute()

# The extension base path
EXT_PATH = Path(extensions_dir).absolute()
BUILTIN_EXT_PATH = Path(extensions_builtin_dir).absolute()

# Tags base path
TAGS_PATH = Path(scripts.basedir()).joinpath("tags").absolute()

# Embeddings directory (Forge Neo: models/embeddings/)
EMB_PATH = Path(shared.cmd_opts.embeddings_dir).absolute()

# Hypernetworks are removed in Forge Neo — always None
HYP_PATH = None

# LoRA and LyCORIS share one directory in Forge Neo
try:
    LORA_PATH = Path(shared.cmd_opts.lora_dir).absolute()
except (AttributeError, TypeError):
    LORA_PATH = None

# LyCORIS is unified with LoRA in Forge Neo
LYCO_PATH = LORA_PATH

# Wildcards directory (Forge Neo: scripts/wildcards/ under webui root)
WILDCARD_PATH = FILE_DIR.joinpath("scripts/wildcards").absolute()


def find_ext_wildcard_paths():
    """Returns paths to wildcard folders registered by other extensions."""
    found = list(EXT_PATH.glob("*/wildcards/"))
    found.extend(BUILTIN_EXT_PATH.glob("*/wildcards/"))

    # Append custom wildcard path from sd-dynamic-prompts if present
    try:
        from modules.shared import opts as _opts
        custom = getattr(_opts, "wildcard_dir", None)
    except ImportError:
        custom = None

    if custom is not None:
        p = Path(custom).absolute()
        if p.exists():
            found.append(p)

    return found


# The path to the extension wildcards folder
WILDCARD_EXT_PATHS = find_ext_wildcard_paths()

# Temporary file paths
STATIC_TEMP_PATH = Path(data_path).joinpath("tmp").absolute()
STATE_PATH = Path(data_path).joinpath(
    "extension-data", "sd-webui-tagcomplete-neo"
).absolute()
TEMP_PATH = STATE_PATH.joinpath("temp")
TAG_FREQUENCY_DB_PATH = STATE_PATH.joinpath("tag_frequency.db")

# Preserve frequency history from pre-built-in installs without removing the
# legacy file. Cache files are disposable and are regenerated in STATE_PATH.
legacy_frequency_db = TAGS_PATH.joinpath("tag_frequency.db")
STATE_PATH.mkdir(parents=True, exist_ok=True)
if not TAG_FREQUENCY_DB_PATH.exists() and legacy_frequency_db.is_file():
    import shutil
    try:
        shutil.copy2(legacy_frequency_db, TAG_FREQUENCY_DB_PATH)
    except OSError as e:
        print(f"Tag Autocomplete: frequency database migration failed: {e}")

# Make sure these folders exist.
# Use parents=True + exist_ok=True so the extension keeps working after a Forge
# update or reinstall that might briefly remove the tags/temp subdirectory (#302).
TEMP_PATH.mkdir(parents=True, exist_ok=True)
STATIC_TEMP_PATH.mkdir(parents=True, exist_ok=True)
