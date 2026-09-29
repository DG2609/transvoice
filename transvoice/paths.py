"""Filesystem layout and audio constants shared by the app and the benchmark."""
import os
import sys
from pathlib import Path


def _user_data_dir() -> Path:
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "TransVoice"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "TransVoice"
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "transvoice"


def _writable(folder: Path) -> bool:
    try:
        probe = folder / ".transvoice-write-test"
        probe.write_text("")
        probe.unlink()
        return True
    except OSError:
        return False


def _root() -> Path:
    """Where models/ and sessions/ live. Source checkout: the repo root. Packaged app: next to the executable
    (portable: unzip and run), unless that folder is read-only (the .deb's /opt, a mounted .dmg): then the
    user's data folder. TRANSVOICE_HOME overrides both."""
    if os.environ.get("TRANSVOICE_HOME"):
        return Path(os.environ["TRANSVOICE_HOME"])
    if not getattr(sys, "frozen", False):
        return Path(__file__).resolve().parents[1]
    here = Path(sys.executable).parent
    if (here / "models").is_dir() or _writable(here):
        return here
    return _user_data_dir()


ROOT = _root()
MODELS = ROOT / "models"
SAMPLE_RATE = 16000
