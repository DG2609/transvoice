"""How should long speech be cut into chunks? Chunk-level ASR error for several cutting rules, on FLEURS
(read speech, gold text) and on recorded clips (reference transcripts). Chunks are cut exactly as live:
AutoGain, Silero VAD, soft/hard limits; each chunk is recognised on its own (as drafts are).

  python scripts/sweep_cuts.py --fleurs 60 --clip .cache/clips/ja_news.wav@.cache/clips/ja_news.ref.txt --cpus 0-5
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from bench.data import fleurs_utterances  # noqa: E402
from bench.metrics import asr_error_rate  # noqa: E402
from bench.resources import limit_cpus  # noqa: E402
from transvoice.asr import REGISTRY as ASR  # noqa: E402
from transvoice.audio import AutoGain  # noqa: E402
from transvoice.engine import Engine, Settings  # noqa: E402
from transvoice.textnorm import asr_text_for_mt  # noqa: E402

RULES = {  # name -> (quiet seconds, ramp towards 64 ms, start of the ramp as a fraction of soft..hard)
    "fixed-64": (0.064, False, 0.0),
    "fixed-192": (0.192, False, 0.0),
    "ramp-192": (0.192, True, 0.0),
    "ramp-256": (0.256, True, 0.0),
    "late-192": (0.192, True, 0.5),
    "late-256": (0.256, True, 0.5),
}


def cut(audio: np.ndarray, quiet_s: float, ramp: bool, ramp_from: float) -> list[tuple[np.ndarray, bool]]:
    engine = Engine(Settings(), lambda *_: None)
    cv = engine._vad_for("them")
    cv.ramp, cv.ramp_from = ramp, ramp_from
    cv.set_quiet(quiet_s)
    agc, out = AutoGain(), []
    for i in range(0, len(audio), 1600):
        out += cv.accept(agc(audio[i:i + 1600]))
    out += cv.flush()
    return [(c, f) for c, f, _ in out if len(c) >= 3200]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fleurs", type=int, default=60)
    ap.add_argument("--clip", action="append", default=[], help="wav@reference.txt")
    ap.add_argument("--rules", default=",".join(RULES))
    ap.add_argument("--cpus")
    args = ap.parse_args()
    limit_cpus(args.cpus)
    asr = ASR["parakeet-ja"].build("ja", 2)
    utts = list(fleurs_utterances("ja").values())[: args.fleurs]
    fleurs = [(np.concatenate([u.load(), np.zeros(8000, np.float32)]), u.text) for u in utts]
    clips = []
    for spec in args.clip:
        wav, ref = spec.split("@", 1)
        audio, _ = sf.read(wav, dtype="float32")
        text = "".join(re.sub(r"^\d+:\d+\s*", "", line) for line in Path(ref).read_text(encoding="utf-8").splitlines()
                       if line and not line.startswith("#") and "(" not in line)
        clips.append((Path(wav).stem, audio, text))
    for name in args.rules.split(","):
        quiet_s, ramp, ramp_from = RULES[name]
        refs, hyps, hard, soft = [], [], 0, 0
        for audio, gold in fleurs:
            cs = cut(audio, quiet_s, ramp, ramp_from)
            hard += sum(f and len(c) >= 4.9 * 16000 for c, f in cs)
            soft += sum(f and len(c) < 4.9 * 16000 for c, f in cs)
            refs.append(gold)
            hyps.append("".join(asr_text_for_mt(asr.transcribe(c)) for c, _ in cs))
        line = (f"{name:10} FLEURS ja {len(refs)} utt: CER {asr_error_rate(refs, hyps, 'ja'):.2f}% "
                f"(soft cuts {soft}, hard cuts {hard})")
        for clip_name, audio, text in clips:
            cs = cut(audio, quiet_s, ramp, ramp_from)
            hyp = "".join(asr_text_for_mt(asr.transcribe(c)) for c, _ in cs)
            line += (f" | {clip_name}: CER {asr_error_rate([text], [hyp], 'ja'):.2f}% "
                     f"({sum(f and len(c) >= 4.9 * 16000 for c, f in cs)} hard cuts)")
        print(line, flush=True)


if __name__ == "__main__":
    main()
