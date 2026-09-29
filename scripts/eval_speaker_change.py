"""Can a speaker embedding tell that the next chunk comes from another speaker?

Chunks are cut as live (AutoGain + VAD + soft/hard limits) from FLEURS recordings (ja, en, vi):
- same speaker: the sentence so far vs its next chunk, inside one recording;
- another speaker: the end of one recording vs the first chunk of another (different gender = surely another
  person; same gender = most likely another person).
Prints the cosine-similarity distributions and, per threshold, how many same-speaker joins would be split and
how many speaker changes would be caught.

  python scripts/eval_speaker_change.py --n 40 --cpus 0,1,2,3
"""
import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import sherpa_onnx  # noqa: E402

from bench.data import DATA, FLEURS_DIRS, _fleurs_tsv  # noqa: E402
from bench.resources import limit_cpus  # noqa: E402
from transvoice.audio import AutoGain  # noqa: E402
from transvoice.engine import Engine, Settings  # noqa: E402
from transvoice.paths import MODELS  # noqa: E402

import soundfile as sf  # noqa: E402

MODELS_SPK = {
    "campplus-zh-en": "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx",
    "wespeaker-campp": "wespeaker_en_voxceleb_CAM++_LM.onnx",
}


def chunks(audio: np.ndarray, lang: str) -> list[np.ndarray]:
    engine = Engine(Settings(), lambda *_: None)
    engine.their_last = lang
    cv = engine._vad_for("them")
    agc, out = AutoGain(), []
    for i in range(0, len(audio), 1600):
        out += cv.accept(agc(audio[i:i + 1600]))
    out += cv.flush()
    return [(c, gap) for c, _, gap in out if len(c) >= 8000]  # >= 0.5 s; gap = quiet at the cut before it


def recordings(lang: str, n: int, seed: int = 3) -> list[tuple[str, list[np.ndarray]]]:
    df = _fleurs_tsv(lang)
    rows = df.sample(n=min(n, len(df)), random_state=seed)
    out = []
    for _, r in rows.iterrows():
        audio, sr = sf.read(DATA / "fleurs" / FLEURS_DIRS[lang] / "test" / r["file"], dtype="float32")
        cs = chunks(audio, lang)
        if cs:
            out.append((str(r.get("gender", "")), [c for c, _ in cs], [g for _, g in cs]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="recordings per language")
    ap.add_argument("--langs", default="ja,en,vi")
    ap.add_argument("--models", default="campplus-zh-en")
    ap.add_argument("--cpus")
    args = ap.parse_args()
    limit_cpus(args.cpus)

    data = {lang: recordings(lang, args.n) for lang in args.langs.split(",")}
    for name in args.models.split(","):
        cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(MODELS / "spk" / MODELS_SPK[name]), num_threads=1)
        ex = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
        spent, calls = 0.0, 0

        def emb(x: np.ndarray) -> np.ndarray:
            nonlocal spent, calls
            t0 = time.perf_counter()
            s = ex.create_stream()
            s.accept_waveform(16000, x)
            s.input_finished()
            v = np.array(ex.compute(s))
            spent += time.perf_counter() - t0
            calls += 1
            return v / (np.linalg.norm(v) + 1e-9)

        same, diff_gender, same_gender = [], [], []  # (similarity, shorter side in s, gap before the chunk)
        for lang, recs in data.items():
            embs = [[emb(c) for c in cs] for _, cs, _ in recs]
            for (_, cs, gaps), es in zip(recs, embs):
                for i in range(1, len(cs)):
                    ctx_audio = np.concatenate(cs[max(0, i - 2):i])[-6 * 16000:]
                    ctx = es[i - 1] if i == 1 else emb(ctx_audio)
                    same.append((float(ctx @ es[i]), min(len(ctx_audio), len(cs[i])) / 16000, gaps[i - 1]))
            rng = random.Random(1)
            for a in range(len(recs)):
                b = rng.randrange(len(recs))
                if b == a:
                    continue
                s = float(embs[a][-1] @ embs[b][0])
                item = (s, min(len(recs[a][1][-1]), len(recs[b][1][0])) / 16000, float("inf"))
                (diff_gender if recs[a][0] != recs[b][0] else same_gender).append(item)
        print(f"== {name}: {1000 * spent / max(1, calls):.0f} ms per embedding (1 thread)")
        for min_s in (0.5, 1.0, 1.5):
            for pause_only in (False, True):
                ok = lambda x: x[1] >= min_s and (not pause_only or x[2] >= 0.19)  # noqa: E731
                sv = np.array([x[0] for x in same if ok(x)])
                dg = np.array([x[0] for x in diff_gender if ok(x)])
                sg = np.array([x[0] for x in same_gender if ok(x)])
                print(f"   chunks >= {min_s} s{', joins at a real pause only' if pause_only else ''}: "
                      f"n same {len(sv)}, other-gender {len(dg)}, same-gender {len(sg)}")
                for th in (0.15, 0.2, 0.25, 0.3):
                    print(f"      threshold {th:.2f}: same speaker split {100 * np.mean(sv < th):4.1f}%  "
                          f"other gender caught {100 * np.mean(dg < th):5.1f}%  same gender caught "
                          f"{100 * np.mean(sg < th):5.1f}%")


if __name__ == "__main__":
    main()
