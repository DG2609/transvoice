"""Download models into models/ (see transvoice/download.py for the list).

Usage:  python scripts/download_models.py [name ...]    (no args = everything, benchmark included)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transvoice.download import MANIFEST, download  # noqa: E402

if __name__ == "__main__":
    download(sys.argv[1:] or list(MANIFEST))
