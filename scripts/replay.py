"""Replay a recording through the app in real time on a CPU subset, with Settings overrides.

  python scripts/replay.py .cache/youtube_run2.wav --cpus 12,13,14,15 --set sentence_max_s=8 > out.txt

Prints the same console output as `python -m transvoice --file ... --realtime --no-overlay`.
"""
import argparse
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.resources import limit_cpus  # noqa: E402
import transvoice.engine as engine  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", type=Path)
    ap.add_argument("--cpus")
    ap.add_argument("--me", default="vi")
    ap.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE")
    args = ap.parse_args()
    limit_cpus(args.cpus)

    overrides = {k: int(v) if v.isdigit() else float(v) for k, v in (item.split("=", 1) for item in args.set)}
    original = engine.Engine.__init__

    engines = []

    def init(self, settings, on_event):
        for k, v in overrides.items():
            setattr(settings, k, v)
        original(self, settings, on_event)
        engines.append(self)

    engine.Engine.__init__ = init
    sys.argv = ["transvoice", "--file", str(args.audio), "--realtime", "--no-overlay", "--me", args.me]
    try:
        runpy.run_module("transvoice", run_name="__main__")
    finally:
        for e in engines:  # translator time: drafts, finals, cancelled work
            print("mt_stats", {k: round(v, 1) for k, v in sorted(e.mt_stats.items())}, flush=True)


if __name__ == "__main__":
    main()
