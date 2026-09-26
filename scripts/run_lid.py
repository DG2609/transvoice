"""Spoken language identification for the JA/EN/VI router: accuracy vs. how much audio it hears.

Candidates:
  whisper-tiny / whisper-base  sherpa-onnx Whisper LID (99 languages; anything else counts as wrong)
  sensevoice                   language tag SenseVoice emits while transcribing; it knows ja/en but not vi,
                               so for a JA/EN/VI app any other tag is mapped to "vi"
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import sherpa_onnx  # noqa: E402

from transvoice.asr import _sense_voice  # noqa: E402
from bench.data import ROOT, SAMPLE_RATE, fleurs_asr  # noqa: E402

LID_DIR = ROOT / "models" / "lid"
OUT = ROOT / "results" / "lid"


def whisper_lid(size: str, threads: int):
    d = LID_DIR / f"whisper-{size}"
    cfg = sherpa_onnx.SpokenLanguageIdentificationConfig(
        whisper=sherpa_onnx.SpokenLanguageIdentificationWhisperConfig(
            encoder=str(d / f"{size}-encoder.int8.onnx"), decoder=str(d / f"{size}-decoder.int8.onnx")),
        num_threads=threads,
    )
    slid = sherpa_onnx.SpokenLanguageIdentification(cfg)

    def detect(samples):
        s = slid.create_stream()
        s.accept_waveform(SAMPLE_RATE, samples)
        return slid.compute(s)

    return detect


def sensevoice_lid(threads: int):
    asr = _sense_voice(threads, "auto")

    def detect(samples):
        s = asr.rec.create_stream()
        s.accept_waveform(SAMPLE_RATE, samples)
        asr.rec.decode_stream(s)
        tag = s.result.lang.strip("<|>")
        return tag if tag in ("ja", "en") else "vi"

    return detect


def trim_leading_silence(samples: np.ndarray, frame_s: float = 0.02, rel_threshold: float = 0.05) -> np.ndarray:
    """Drop audio before the first 20 ms frame whose RMS reaches 5% of the loudest frame (a stand-in for VAD)."""
    n = int(frame_s * SAMPLE_RATE)
    frames = samples[: len(samples) // n * n].reshape(-1, n)
    rms = np.sqrt((frames**2).mean(axis=1))
    loud = np.nonzero(rms >= rel_threshold * rms.max())[0]
    return samples[loud[0] * n :] if len(loud) else samples


DETECTORS = {
    "whisper-tiny": lambda t: whisper_lid("tiny", t),
    "whisper-base": lambda t: whisper_lid("base", t),
    "sensevoice": sensevoice_lid,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seconds", default="1,2,3,full")
    args = ap.parse_args()

    utts = {lang: fleurs_asr(lang, args.n, seed=1) for lang in ("ja", "en", "vi")}
    # In the app, VAD starts the clip at speech onset, so strip the silence FLEURS recordings begin with.
    audio = {lang: [trim_leading_silence(u.load()) for u in us] for lang, us in utts.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for name, build in DETECTORS.items():
        detect = build(args.threads)
        for sec in args.seconds.split(","):
            n_samples = None if sec == "full" else int(float(sec) * SAMPLE_RATE)
            correct, total, times, confusion = 0, 0, [], Counter()
            per_lang = {}
            for lang, clips in audio.items():
                ok = 0
                for a in clips:
                    clip = a[:n_samples] if n_samples else a[: 30 * SAMPLE_RATE]
                    t0 = time.perf_counter()
                    pred = detect(clip)
                    times.append(time.perf_counter() - t0)
                    ok += pred == lang
                    confusion[f"{lang}->{pred}"] += 1
                per_lang[lang] = round(100 * ok / len(clips), 1)
                correct += ok
                total += len(clips)
            times.sort()
            r = {"detector": name, "audio": sec, "accuracy": round(100 * correct / total, 1), **per_lang,
                 "p50_ms": round(1000 * times[len(times) // 2]), "errors": {k: v for k, v in confusion.items()
                                                                              if k.split("->")[0] != k.split("->")[1]}}
            results.append(r)
            print(json.dumps(r, ensure_ascii=False), flush=True)
    (OUT / "summary.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results),
                                      encoding="utf-8")


if __name__ == "__main__":
    main()
