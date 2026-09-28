"""Compare ASR models on a real recording, cut into the same chunks the app uses.

  python scripts/compare_asr_clip.py .cache/clips/ja_news.wav --lang ja --models parakeet-ja,reazonspeech-k2
      [--ref .cache/clips/ja_news.ref.txt]

Prints each chunk's text per model, then (with --ref) the CER of the joined text against the reference.
Chunks are cut exactly as live: Silero VAD + the soft/hard length limits of engine.Settings.
"""
import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from bench.metrics import asr_error_rate  # noqa: E402
from transvoice.asr import REGISTRY as ASR  # noqa: E402
from transvoice.engine import Engine, Settings  # noqa: E402
from transvoice.textnorm import asr_text_for_mt  # noqa: E402


def chunks(audio: np.ndarray, settings: Settings, quiet_windows: int) -> list[tuple[np.ndarray, bool]]:
    engine = Engine(settings, lambda *_: None)
    cv = engine._vad_for("them")
    cv.quiet_windows = quiet_windows  # overrides settings.chunk_quiet_s
    out = []
    step = 1600
    for i in range(0, len(audio), step):
        out += cv.accept(audio[i:i + step])
    out += cv.flush()
    return [(c, f) for c, f, _ in out if len(c) >= 3200]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", type=Path)
    ap.add_argument("--lang", default="ja")
    ap.add_argument("--models", default="parakeet-ja,reazonspeech-k2")
    ap.add_argument("--ref", type=Path)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--quiet-windows", type=int, default=2, help="32 ms windows of quiet needed for a soft cut")
    ap.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE")
    ap.add_argument("--quiet", action="store_true", help="print only the summary")
    args = ap.parse_args()
    settings = Settings(**{k: float(v) for k, v in (i.split("=", 1) for i in args.set)})

    audio, sr = sf.read(args.audio, dtype="float32")
    assert sr == 16000
    cs = chunks(audio, settings, args.quiet_windows)
    print(f"{len(cs)} chunks ({sum(f for _, f in cs)} cut mid-speech), {len(audio) / sr:.0f} s audio", flush=True)
    names = args.models.split(",")
    texts = {}
    for name in names:
        asr = ASR[name].build(args.lang, args.threads)
        t0 = time.perf_counter()
        texts[name] = [asr_text_for_mt(asr.transcribe(c)) for c, _ in cs]
        print(f"{name}: {time.perf_counter() - t0:.1f} s compute", flush=True)
    for i in range(0 if not args.quiet else len(cs), len(cs)):
        print(f"--- chunk {i} ({len(cs[i][0]) / 16000:.1f} s{', forced' if cs[i][1] else ''})")
        for name in names:
            print(f"  {name:16} {texts[name][i]}")
    if args.ref:
        ref = "".join(re.sub(r"^\d+:\d+\s*", "", line) for line in args.ref.read_text(encoding="utf-8").splitlines()
                      if line and not line.startswith("#") and "(" not in line)
        for name in names:
            hyp = "".join(texts[name])
            print(f"CER vs reference {name}: {asr_error_rate([ref], [hyp], args.lang):.2f}%")


if __name__ == "__main__":
    main()
