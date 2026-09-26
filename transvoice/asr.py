"""ASR candidates, all run through sherpa-onnx on CPU."""
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import sherpa_onnx

from .paths import ROOT, SAMPLE_RATE

ASR_DIR = ROOT / "models" / "asr"
NEMOTRON_LANG = {"ja": "ja-JP", "en": "en-US", "vi": "vi-VN"}
NEMOTRON_CHUNK_S = 0.56


def _pkg(prefix: str) -> Path:
    found = sorted(ASR_DIR.glob(prefix + "*"))
    if not found:
        raise FileNotFoundError(f"model package {prefix}* not found in {ASR_DIR}")
    return found[-1]


def _one(d: Path, pattern: str) -> str:
    found = sorted(d.glob(pattern))
    if len(found) != 1:
        raise FileNotFoundError(f"expected one {pattern} in {d}, found {[f.name for f in found]}")
    return str(found[0])


class OfflineAsr:
    streaming = False

    def __init__(self, recognizer):
        self.rec = recognizer

    def transcribe(self, samples: np.ndarray) -> str:
        s = self.rec.create_stream()
        s.accept_waveform(SAMPLE_RATE, samples)
        self.rec.decode_stream(s)
        return s.result.text.strip()


@dataclass
class StreamingAsr:
    """Feeds audio in 100 ms pieces like a live mic and times each chunk decode."""

    rec: object
    language: str
    streaming: bool = True
    chunk_ms: list[float] = field(default_factory=list)

    def transcribe(self, samples: np.ndarray) -> str:
        s = self.rec.create_stream()
        if self.language:
            s.set_option("language", self.language)
        piece = SAMPLE_RATE // 10
        tail = np.zeros(int((NEMOTRON_CHUNK_S + 0.1) * SAMPLE_RATE), dtype=np.float32)
        for chunk in [*(samples[i : i + piece] for i in range(0, len(samples), piece)), tail]:
            s.accept_waveform(SAMPLE_RATE, chunk)
            while self.rec.is_ready(s):
                t0 = time.perf_counter()
                self.rec.decode_stream(s)
                self.chunk_ms.append(1000 * (time.perf_counter() - t0))
        s.input_finished()
        while self.rec.is_ready(s):
            self.rec.decode_stream(s)
        return self.rec.get_result(s).strip()


def _transducer(prefix: str, threads: int, **kw) -> OfflineAsr:
    d = _pkg(prefix)
    return OfflineAsr(
        sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=_one(d, "encoder*.int8.onnx"),
            decoder=_one(d, "decoder*.onnx"),
            joiner=_one(d, "joiner*.int8.onnx"),
            tokens=_one(d, "tokens.txt"),
            num_threads=threads,
            **kw,
        )
    )


def _sense_voice(threads: int, language: str) -> OfflineAsr:
    d = _pkg("sherpa-onnx-sense-voice-")
    return OfflineAsr(
        sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=_one(d, "model.int8.onnx"),
            tokens=_one(d, "tokens.txt"),
            num_threads=threads,
            language=language,
            use_itn=True,
        )
    )


def _qwen3(threads: int) -> OfflineAsr:
    d = _pkg("sherpa-onnx-qwen3-asr-")
    return OfflineAsr(
        sherpa_onnx.OfflineRecognizer.from_qwen3_asr(
            conv_frontend=_one(d, "conv_frontend.onnx"),
            encoder=_one(d, "encoder*.onnx"),
            decoder=_one(d, "decoder*.onnx"),
            tokenizer=str(d / "tokenizer"),
            num_threads=threads,
        )
    )


def _parakeet_ja(threads: int) -> OfflineAsr:
    d = _pkg("sherpa-onnx-nemo-parakeet-tdt_ctc-0.6b-ja")
    return OfflineAsr(
        sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
            model=_one(d, "model.int8.onnx"), tokens=_one(d, "tokens.txt"), num_threads=threads
        )
    )


def _nemotron(threads: int, language: str) -> StreamingAsr:
    d = _pkg("sherpa-onnx-nemotron-3.5-asr-streaming-")
    rec = sherpa_onnx.OnlineRecognizer.from_transducer(
        encoder=_one(d, "encoder.int8.onnx"),
        decoder=_one(d, "decoder.int8.onnx"),
        joiner=_one(d, "joiner.int8.onnx"),
        tokens=_one(d, "tokens.txt"),
        num_threads=threads,
        provider="cpu",
    )
    return StreamingAsr(rec, language)


@dataclass
class AsrSpec:
    langs: tuple[str, ...]
    build: Callable[[str, int], object]  # (lang, threads) -> transcriber
    note: str = ""


REGISTRY: dict[str, AsrSpec] = {
    "sensevoice": AsrSpec(("ja", "en"), lambda lang, t: _sense_voice(t, lang), "234M, language forced"),
    "sensevoice-auto": AsrSpec(("ja", "en"), lambda lang, t: _sense_voice(t, "auto"), "language auto-detected"),
    "zipformer-vi-30m": AsrSpec(("vi",), lambda lang, t: _transducer("sherpa-onnx-zipformer-vi-30M-", t), "30M"),
    "vietasr": AsrSpec(("vi",), lambda lang, t: _transducer("sherpa-onnx-zipformer-vi-int8-", t), "68M"),
    "reazonspeech-k2": AsrSpec(("ja",), lambda lang, t: _transducer("reazonspeech-k2-v2", t), "159M"),
    "parakeet-ja": AsrSpec(("ja",), lambda lang, t: _parakeet_ja(t), "600M"),
    # Moonshine base ja (2026-02-27) is excluded: with sherpa-onnx 1.13.8 its decoder raises an
    # ONNX broadcast error on most inputs and returns empty text.
    "qwen3-asr": AsrSpec(("ja", "en", "vi"), lambda lang, t: _qwen3(t), "0.6B, auto language"),
    "nemotron": AsrSpec(("ja", "en", "vi"), lambda lang, t: _nemotron(t, NEMOTRON_LANG[lang]), "0.6B streaming, language forced"),
    "nemotron-auto": AsrSpec(("ja", "en", "vi"), lambda lang, t: _nemotron(t, "auto"), "0.6B streaming, auto language"),
}
