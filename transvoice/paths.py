"""Filesystem layout and audio constants shared by the app and the benchmark."""
import sys
from pathlib import Path

# Source checkout: the repo root. Packaged app (PyInstaller): the folder that holds TransVoice.exe,
# so models/ and sessions/ sit next to the executable.
ROOT = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
SAMPLE_RATE = 16000
