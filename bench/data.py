"""Benchmark datasets: FLEURS (ja/en/vi, parallel sentences) and the ReazonSpeech test split."""
import io
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SAMPLE_RATE = 16000
FLEURS_DIRS = {"ja": "ja_jp", "en": "en_us", "vi": "vi_vn"}
_TSV_COLS = ["id", "file", "raw", "norm", "chars", "num_samples", "gender"]


@dataclass
class Utterance:
    key: str
    lang: str
    text: str
    duration: float
    load: Callable[[], np.ndarray]  # 16 kHz mono float32


def _to_mono_16k(audio: np.ndarray, sr: int) -> np.ndarray:
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        n = int(round(len(audio) * SAMPLE_RATE / sr))
        audio = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio)
    return audio.astype(np.float32)


def _read_file(path: Path) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32")
    return _to_mono_16k(audio, sr)


def _read_bytes(data: bytes) -> np.ndarray:
    audio, sr = sf.read(io.BytesIO(data), dtype="float32")
    return _to_mono_16k(audio, sr)


def _fleurs_tsv(lang: str) -> pd.DataFrame:
    d = FLEURS_DIRS[lang]
    df = pd.read_csv(DATA / "fleurs" / d / "test.tsv", sep="\t", header=None, names=_TSV_COLS, quoting=3)
    return df.sort_values(["id", "file"])


def fleurs_utterances(lang: str) -> dict[int, Utterance]:
    """FLEURS sentence id -> its first recording."""
    df = _fleurs_tsv(lang).drop_duplicates("id")
    audio_dir = DATA / "fleurs" / FLEURS_DIRS[lang] / "test"
    return {
        int(r["id"]): Utterance(
            key=f"fleurs-{lang}-{r['id']}",
            lang=lang,
            text=r["raw"],
            duration=r["num_samples"] / SAMPLE_RATE,
            load=lambda p=audio_dir / r["file"]: _read_file(p),
        )
        for r in df.to_dict("records")
    }


def fleurs_asr(lang: str, n: int, seed: int = 0) -> list[Utterance]:
    """One recording per FLEURS sentence, `n` sentences sampled deterministically."""
    utts = list(fleurs_utterances(lang).values())
    return random.Random(seed).sample(utts, min(n, len(utts)))


def reazonspeech_asr(n: int, seed: int = 0) -> list[Utterance]:
    files = sorted((DATA / "reazonspeech_test").glob("*.parquet"))
    rows = []
    for f in files:
        rows.extend(pq.read_table(f).to_pylist())
    idx = random.Random(seed).sample(range(len(rows)), min(n, len(rows)))
    out = []
    for i in idx:
        audio = _read_bytes(rows[i]["audio"]["bytes"])
        out.append(
            Utterance(
                key=f"reazon-{i}",
                lang="ja",
                text=rows[i]["transcription"],
                duration=len(audio) / SAMPLE_RATE,
                load=lambda a=audio: a,
            )
        )
    return out


def fleurs_parallel(n: int, seed: int = 0) -> list[dict]:
    """Sentences present in all three FLEURS test sets: [{"id", "ja", "en", "vi"}]."""
    texts = {}
    for lang in FLEURS_DIRS:
        df = _fleurs_tsv(lang).drop_duplicates("id")
        texts[lang] = dict(zip(df["id"], df["raw"]))
    common = sorted(set.intersection(*(set(t) for t in texts.values())))
    chosen = random.Random(seed).sample(common, min(n, len(common)))
    return [{"id": int(i), **{lang: texts[lang][i] for lang in FLEURS_DIRS}} for i in chosen]


def asr_dataset(name: str, lang: str, n: int) -> list[Utterance]:
    if name == "fleurs":
        return fleurs_asr(lang, n)
    if name == "reazon":
        assert lang == "ja", "ReazonSpeech is Japanese only"
        return reazonspeech_asr(n)
    raise ValueError(name)
