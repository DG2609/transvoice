"""Speech -> text -> translation pipeline (CPU only), live-subtitle style.

Two audio channels: "them" (system audio: the call/video you are listening to) and "me" (microphone).

Audio -> Silero VAD cut into short chunks (2-4 s, cut in pauses) -> language -> ASR -> the chunk is
appended to the channel's open *sentence*. The translator re-translates the whole sentence-so-far every
time it grows, so the subtitle updates chunk by chunk while the model always sees the full sentence
context (Japanese puts the verb and negation last). When the sentence ends, the last translation is
final. If translation falls behind, intermediate drafts are skipped: only the newest text is translated.

Threads: VAD, ASR and MT run separately so recognition of the next chunk overlaps translation.
"""
import collections
import itertools
import queue
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import sherpa_onnx

from .asr import REGISTRY as ASR
from .audio import AutoGain
from .mt import LlamaEngine, MT_DIR, TranslationCancelled, make_hy_mt_prompt
from .paths import MODELS, SAMPLE_RATE
from .textnorm import asr_text_for_mt, normalize

LANGS = ("ja", "en", "vi")
DEFAULT_ASR = {"ja": "parakeet-ja", "en": "sensevoice", "vi": "zipformer-vi-30m"}
DEFAULT_MT = MT_DIR / "HY-MT1.5-1.8B-Q4_K_M.gguf"


@dataclass
class Settings:
    my_lang: str = "vi"
    their_lang: str = "auto"  # "auto" = detect among `langs`
    langs: tuple[str, ...] = LANGS
    asr: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_ASR))
    mt_model: str = str(DEFAULT_MT)
    glossary: list[dict] = field(default_factory=list)
    asr_threads: int = 2
    mt_threads: int = 4
    low_priority: bool = True
    vad_min_silence: float = 0.4  # silence that ends a chunk
    # Chunk length: after `soft` seconds of speech cut at the next quiet 32 ms window, after `hard` cut anyway.
    chunk_soft_s: float = 2.5
    chunk_hard_s: float = 5.0
    sentence_gap_s: float = 0.8  # extra pause after a chunk that closes the sentence
    sentence_max_s: float = 12.0  # close long sentences so the final re-translation stays fast
    drafts: bool = True  # translate unfinished sentences (live feel); False = only finished sentences
    # When a sentence ends, recognise its whole audio again: chunk cuts garble words at the joins
    # ("一番一番は", "金金属"), and the final translation should not inherit that.
    rescore_final: bool = True
    rescore_max_s: float = 10.0  # longer sentences cost too much ASR time on office CPUs (p90 lag 8 -> 15 s)
    # Faster re-translation: reuse the KV cache of the shared prompt prefix and draft tokens with n-gram
    # lookup. ~18% faster on 4 E-cores, but outputs are not bit-identical to the plain run.
    mt_fast: bool = True
    lid_min_seconds: float = 1.0  # shorter first chunks reuse the last detected language


@dataclass
class Sentence:
    id: int
    channel: str  # "them" | "me"
    lang: str
    target: str
    started_at: float
    chunks: list[str] = field(default_factory=list)
    last_chunk_at: float = 0.0  # wall clock when the last chunk's audio ended
    last_forced: bool = False  # the last chunk was cut mid-speech, so the speaker is still talking
    lang_confirmed: bool = True  # False while the language is a guess from too little audio
    shared_audio: bool = False  # split inside a chunk: its audio also holds part of a neighbouring sentence
    rescoring: bool = False  # closed, whole-sentence recognition still running
    audio_s: float = 0.0
    closed: bool = False
    version: int = 0  # bumps whenever a chunk is appended
    translated_version: int = 0
    translation: str | None = None
    timings: dict = field(default_factory=dict)

    @property
    def text(self) -> str:
        return ("" if self.lang == "ja" else " ").join(self.chunks)

    @property
    def needs_translation(self) -> bool:
        return self.target != self.lang and self.version > self.translated_version

    @property
    def final(self) -> bool:
        return self.closed and not self.needs_translation

    def snapshot(self) -> "Sentence":
        return Sentence(**{**self.__dict__, "chunks": list(self.chunks), "timings": dict(self.timings)})


class LanguageId:
    """Whisper-base spoken-language ID."""

    def __init__(self, threads: int):
        d = MODELS / "lid" / "whisper-base"
        cfg = sherpa_onnx.SpokenLanguageIdentificationConfig(
            whisper=sherpa_onnx.SpokenLanguageIdentificationWhisperConfig(
                encoder=str(d / "base-encoder.int8.onnx"), decoder=str(d / "base-decoder.int8.onnx")),
            num_threads=threads,
        )
        self.slid = sherpa_onnx.SpokenLanguageIdentification(cfg)

    def detect(self, samples: np.ndarray) -> str:
        s = self.slid.create_stream()
        s.accept_waveform(SAMPLE_RATE, samples[: 30 * SAMPLE_RATE])
        return self.slid.compute(s)


# Single words the ASR models produce from noise, music or breathing; alone they are never worth a subtitle.
FILLERS = {
    "en": {"the", "a", "i", "and", "uh", "um", "oh", "hmm", "mm"},
    "ja": {"ピ", "ん", "うん", "え", "あ", "あっ", "えー", "はあ"},
    "vi": {"và", "tôi", "à", "ờ", "ừ", "ừm", "thì"},
}
_SENTENCE_END = {
    "en": re.compile(r"[.?!]\W*$"),
    "ja": re.compile(r"([。？！?!]|(です|ます|ました|でした|ません|ませんでした|ください|でしょう)[かねよ]?)\W*$"),
    "vi": re.compile(r"[.?!]\W*$"),
}


def is_noise(text: str, lang: str, audio_s: float) -> bool:
    """ASR models emit fillers ("The.", "うん", "TÔI") on noise; drop them and tiny outputs from short segments."""
    n = normalize(text, lang)
    if not n or n in FILLERS.get(lang, ()):
        return True
    size = len(n) if lang == "ja" else len(n.split())
    return audio_s < 0.8 and size <= (2 if lang == "ja" else 1)


def ends_sentence(text: str, lang: str) -> bool:
    return bool(_SENTENCE_END.get(lang, _SENTENCE_END["en"]).search(text.strip()))


# Sentence boundaries inside a chunk. Latin text splits at .?! before a capital letter, so "U.S." or "3.5"
# stay intact. The Japanese ASR rarely writes "。", so polite sentence endings count as boundaries too, but
# only when a new sentence follows: "ましたが" / "ですけど" continue the sentence, and the spoken filler
# "ですね、" is not an ending at all.
_LATIN_SPLIT = re.compile(r"(?<=[.?!])\s+(?=[A-ZÀ-Ỹ])")
_JA_BOUNDARY = re.compile(
    r"[。？！?!]"
    r"|(?:ですか|ますか|ですよね|ますよね|でしょうか|ませんでした|ました|でした|ません|ください)"
    r"(?=[^がけしのとばてでねよなかわ、。？！?!\s])"
)


_JA_CONTINUATION = ("が", "けど", "けれど", "し", "から", "ので", "のに", "と", "ね", "よ", "な", "って", "で")


def continues_sentence(piece: str, lang: str) -> bool:
    """Does this chunk continue the previous one even though that one looked finished?"""
    p = piece.lstrip("、, ")
    if lang == "ja":
        return p.startswith(_JA_CONTINUATION)
    return bool(p) and p[0].islower()  # "... ballot. and then" -> the period was an ASR guess


def split_sentences(text: str, lang: str) -> list[str]:
    if lang == "ja":
        cuts = [m.end() for m in _JA_BOUNDARY.finditer(text)]
        parts = [text[a:b] for a, b in zip([0, *cuts], [*cuts, len(text)])]
    else:
        parts = _LATIN_SPLIT.split(text)
    parts = [p.strip() for p in parts if p.strip()]
    return parts or [text]


class _ChannelVad:
    """Silero VAD for one channel plus a chunk-length limit that prefers to cut in a pause between words."""

    WINDOW = 512  # Silero's frame size at 16 kHz

    QUIET_WINDOWS = 2  # a cut needs 64 ms well below the speech level: a gap between words, not a soft syllable

    def __init__(self, vad, soft_s: float, hard_s: float):
        self.vad = vad
        self.soft, self.hard = int(soft_s * SAMPLE_RATE), int(hard_s * SAMPLE_RATE)
        self.pending = np.zeros(0, np.float32)
        self.run = 0  # samples of uninterrupted speech in the open segment
        self.levels: list[float] = []  # RMS of each window in the open segment
        self.quiet = 0  # consecutive quiet windows after the soft limit
        self.last_speech_at = 0.0  # wall clock of the last window the VAD considered speech

    def accept(self, samples: np.ndarray) -> list[tuple[np.ndarray, bool]]:
        """Returns closed segments as (samples, forced); forced = cut mid-speech, the sentence goes on."""
        out = []
        buf = np.concatenate([self.pending, samples])
        n = len(buf) // self.WINDOW * self.WINDOW
        for i in range(0, n, self.WINDOW):
            w = buf[i : i + self.WINDOW]
            self.vad.accept_waveform(w)
            out += self._pop(forced=False)
            if not self.vad.is_speech_detected():
                self.run, self.levels, self.quiet = 0, [], 0
                continue
            self.last_speech_at = time.time()
            self.run += self.WINDOW
            level = float(np.sqrt(np.mean(w * w)))
            self.levels.append(level)
            if self.run >= self.soft and level < 0.3 * float(np.median(self.levels)):
                self.quiet += 1
            else:
                self.quiet = 0
            if self.quiet >= self.QUIET_WINDOWS or self.run >= self.hard:
                self.vad.flush()  # closes the segment here; detection continues with the next window
                self.run, self.levels, self.quiet = 0, [], 0
                out += self._pop(forced=True)
        self.pending = buf[n:]
        return out

    @property
    def speaking(self) -> bool:
        return self.run > 0

    def flush(self) -> list[tuple[np.ndarray, bool]]:
        self.vad.flush()
        self.run, self.levels, self.quiet = 0, [], 0
        return self._pop(forced=False)

    def _pop(self, forced: bool) -> list[tuple[np.ndarray, bool]]:
        out = []
        while not self.vad.empty():
            out.append((np.array(self.vad.front.samples, dtype=np.float32), forced))
            self.vad.pop()
        return out


class Engine:
    def __init__(self, settings: Settings, on_event: Callable[[str, object], None]):
        self.s = settings
        # Events: ("status", str) | ("heard", Sentence) | ("translated", Sentence) | ("error", str).
        # Sentences are snapshots; "translated" with .final=True happens exactly once per sentence.
        self.on_event = on_event
        self.audio_q: queue.Queue = queue.Queue()
        self.seg_q: queue.Queue = queue.Queue()
        self.stop_flag = threading.Event()
        self.vads: dict[str, _ChannelVad] = {}
        self.agc: dict[str, AutoGain] = {}
        self.asr: dict[str, object] = {}
        self.their_last: str | None = None if settings.their_lang == "auto" else settings.their_lang
        self.ids = itertools.count(1)
        self.lid = None
        self.mt = None
        self.lock = threading.Condition()
        self.open: dict[str, Sentence] = {}  # channel -> sentence still growing
        self.sentence_audio: dict[int, list[np.ndarray]] = {}  # chunk audio of sentences not final yet
        self.to_rescore: list[Sentence] = []
        self.mt_job: tuple | None = None  # (sentence id, is draft, version, cancel event) being translated
        self.queued: collections.Counter = collections.Counter()  # chunks waiting for ASR, per channel
        self.pending: dict[int, Sentence] = {}  # sentences that still need translation or a final event
        self.mt_busy = False
        self.threads: list[threading.Thread] = []

    # ---- lifecycle ----
    def start(self) -> None:
        self.on_event("status", "Đang tải model…")
        needed = {self.s.my_lang} | (set(self.s.langs) if self.s.their_lang == "auto" else {self.s.their_lang})
        for lang in sorted(needed):
            self.asr[lang] = ASR[self.s.asr[lang]].build(lang, self.s.asr_threads)
        if self.s.their_lang == "auto":
            self.lid = LanguageId(self.s.asr_threads)
        fast = dict(reuse_prefix=True, extra_args=("--spec-type", "ngram-mod")) if self.s.mt_fast else {}
        self.mt = LlamaEngine(Path(self.s.mt_model), make_hy_mt_prompt(self.s.glossary), self.s.mt_threads,
                              low_priority=self.s.low_priority, **fast)
        self.mt.translate("Hello.", "en", "ja")  # warm-up
        self._start_workers()
        self.on_event("status", "Đang nghe")

    def _start_workers(self) -> None:
        for fn in (self._vad_loop, self._asr_loop, self._mt_loop):
            t = threading.Thread(target=fn, daemon=True)
            t.start()
            self.threads.append(t)

    def feed(self, channel: str, samples: np.ndarray) -> None:
        """Thread-safe; called from audio callbacks."""
        self.audio_q.put((channel, samples))

    def flush(self) -> None:
        """End of input (file mode): close open segments and sentences."""
        self.audio_q.put(("__flush__", None))

    def wait_idle(self, timeout: float = 600) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self.lock:
                busy = self.pending or self.mt_busy
            if not (self.audio_q.unfinished_tasks or self.seg_q.unfinished_tasks or busy):
                return
            time.sleep(0.1)

    def stop(self) -> None:
        self.stop_flag.set()
        with self.lock:
            self.lock.notify_all()
        if self.mt is not None:
            self.mt.close()

    # ---- VAD ----
    def _vad_for(self, channel: str) -> _ChannelVad:
        if channel not in self.vads:
            cfg = sherpa_onnx.VadModelConfig()
            cfg.silero_vad.model = str(MODELS / "vad" / "silero_vad.onnx")
            cfg.silero_vad.min_silence_duration = self.s.vad_min_silence
            cfg.silero_vad.min_speech_duration = 0.25
            cfg.silero_vad.max_speech_duration = self.s.chunk_hard_s + 5  # our own cutter decides
            cfg.sample_rate = SAMPLE_RATE
            cfg.num_threads = 1
            self.vads[channel] = _ChannelVad(sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=60),
                                             self.s.chunk_soft_s, self.s.chunk_hard_s)
        return self.vads[channel]

    def _vad_loop(self) -> None:
        while not self.stop_flag.is_set():
            try:
                channel, samples = self.audio_q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                if channel == "__flush__":
                    for ch, cv in self.vads.items():
                        for seg, forced in cv.flush():
                            self._enqueue(ch, seg, time.time(), forced)
                    self.seg_q.put(("__flush__", None, time.time(), False))
                    continue
                cv = self._vad_for(channel)
                for seg, forced in cv.accept(self.agc.setdefault(channel, AutoGain())(samples)):
                    self._enqueue(channel, seg, time.time(), forced)
            finally:
                self.audio_q.task_done()

    # ---- recognition ----
    LID_CONFIRM_S = 2.0  # language ID on at least this much audio is trusted (~94% at 3 s in the benchmark)

    def _language(self, channel: str, samples: np.ndarray) -> tuple[str, bool]:
        """(language, confirmed). Unconfirmed guesses are re-checked once the sentence has more audio."""
        if channel == "me":
            return self.s.my_lang, True
        if self.s.their_lang != "auto":
            return self.s.their_lang, True
        if len(samples) < self.s.lid_min_seconds * SAMPLE_RATE and self.their_last:
            return self.their_last, False
        lang = self.lid.detect(samples)
        if lang in self.s.langs:
            self.their_last = lang
            return lang, len(samples) >= self.LID_CONFIRM_S * SAMPLE_RATE
        return self.their_last or next(l for l in self.s.langs if l != self.s.my_lang), False

    def _recheck_language(self, s: Sentence, audio: list[np.ndarray]) -> tuple[str, bool, list[str] | None]:
        """Re-run language ID on the whole sentence so far; if it changed, re-recognise earlier chunks."""
        joined = np.concatenate(audio)
        if len(joined) < self.LID_CONFIRM_S * SAMPLE_RATE:
            return s.lang, False, None
        detected = self.lid.detect(joined)
        if detected not in self.s.langs or detected == s.lang:
            return s.lang, True, None
        self.their_last = detected
        texts = [asr_text_for_mt(self.asr[detected].transcribe(a)) for a in audio[:-1]]
        return detected, True, [t for t, a in zip(texts, audio) if not is_noise(t, detected, len(a) / SAMPLE_RATE)]

    def _target(self, channel: str) -> str:
        if channel == "them":
            return self.s.my_lang
        return self.their_last or next(l for l in self.s.langs if l != self.s.my_lang)

    def _enqueue(self, channel: str, samples: np.ndarray, ended_at: float, forced: bool) -> None:
        with self.lock:
            self.queued[channel] += 1
        self.seg_q.put((channel, samples, ended_at, forced))

    def _asr_loop(self) -> None:
        while not self.stop_flag.is_set():
            try:
                channel, samples, ended_at, forced = self.seg_q.get(timeout=0.2)
            except queue.Empty:
                self._close_quiet_sentences()
                self._run_rescores()
                continue
            if channel != "__flush__":
                with self.lock:
                    self.queued[channel] -= 1  # counts chunks still waiting, not the one in hand
            try:
                if channel == "__flush__":
                    with self.lock:
                        for ch in list(self.open):
                            self._close(ch)
                    self._run_rescores()
                    continue
                self._add_chunk(channel, samples, ended_at, forced)
            except Exception as e:  # noqa: BLE001 - keep the app alive, report the failure
                self.on_event("error", f"{type(e).__name__}: {e}")
            finally:
                self.seg_q.task_done()
            self._close_quiet_sentences()
            self._run_rescores()

    def _add_chunk(self, channel: str, samples: np.ndarray, ended_at: float, forced: bool) -> None:
        audio_s = len(samples) / SAMPLE_RATE
        if audio_s < 0.2:  # a forced cut can leave a sliver too short for the ASR front-end (0 frames)
            return
        with self.lock:
            current = self.open.get(channel)
        t0 = time.perf_counter()
        relabeled = None
        if current is None:
            lang, confirmed = self._language(channel, samples)
            audio = [samples]
        else:
            # A sentence keeps its language; a guess made on a short first chunk is re-checked on more audio.
            lang, confirmed = current.lang, current.lang_confirmed
            audio = self.sentence_audio.get(current.id, []) + [samples]
            if not confirmed:
                lang, confirmed, relabeled = self._recheck_language(current, audio)
        t1 = time.perf_counter()
        text = asr_text_for_mt(self.asr[lang].transcribe(samples))
        t2 = time.perf_counter()
        if forced and lang != "ja":
            # SenseVoice ends every chunk with a period, even one cut mid-sentence ("On August. 15");
            # a false period would split the English sentence at every cut.
            text = re.sub(r"[.。]\s*$", "", text)
        noise = is_noise(text, lang, audio_s)
        if noise and relabeled is None:
            return
        with self.lock:
            s = self.open.get(channel)
            if s is None:
                s = self._new_sentence(channel, lang, ended_at - audio_s)
            if relabeled is not None:
                s.lang, s.chunks = lang, relabeled
                s.version += 1
            s.lang_confirmed = confirmed
            # A chunk can span the end of one sentence and the start of the next (a question and its
            # answer); split at explicit sentence punctuation so each sentence is translated on its own.
            pieces = [] if noise else split_sentences(text, lang)
            snaps = []
            for i, piece in enumerate(pieces):
                if s is not None and s.chunks and ends_sentence(s.chunks[-1], lang) \
                        and not continues_sentence(piece, lang) and (lang == "ja" or not s.last_forced):
                    # The previous chunk ended a sentence exactly at the cut ("…思いますか" | "やっぱ…").
                    # Japanese endings are words, so they count even at a forced cut; Latin punctuation at
                    # a forced cut is only an ASR guess.
                    self._close(channel)
                    snaps.append(s.snapshot())
                    s = None
                if s is None:
                    s = self._new_sentence(channel, lang, ended_at - audio_s)
                    s.lang_confirmed = confirmed
                if len(pieces) > 1:
                    s.shared_audio = True
                else:
                    self.sentence_audio.setdefault(s.id, []).append(samples)
                s.chunks.append(piece)
                s.version += 1
                s.audio_s += audio_s * len(piece) / max(1, len(text))
                s.last_chunk_at = ended_at
                s.last_forced = forced
                s.timings.update(lid_ms=round(1000 * (t1 - t0)), asr_ms=round(1000 * (t2 - t1)))
                self.pending[s.id] = s
                last = i == len(pieces) - 1
                too_long = (s.audio_s >= self.s.sentence_max_s and not forced) \
                    or s.audio_s >= 1.25 * self.s.sentence_max_s  # prefer closing at a pause, not mid-word
                if not last or (not forced and ends_sentence(piece, lang)) or too_long:
                    self._close(channel)
                snaps.append(s.snapshot())
                s = None if not last else s
            if not pieces:  # relabel only
                self.pending[s.id] = s
                snaps.append(s.snapshot())
            self.lock.notify_all()
        for snap in snaps:
            self.on_event("heard", snap)

    def _new_sentence(self, channel: str, lang: str, started_at: float) -> Sentence:
        """Caller holds the lock."""
        s = Sentence(next(self.ids), channel, lang, self._target(channel), started_at=started_at)
        self.open[channel] = s
        return s

    def _close(self, channel: str) -> None:
        """Caller holds the lock."""
        s = self.open.pop(channel, None)
        if s is None:
            return
        s.closed = True
        audio = self.sentence_audio.get(s.id, [])
        if (self.s.rescore_final and not s.shared_audio and len(s.chunks) >= 2 and len(audio) >= 2
                and sum(len(a) for a in audio) <= self.s.rescore_max_s * SAMPLE_RATE
                and not self.queued[channel]):  # when ASR is already behind, keep up instead
            s.rescoring = True  # the translator waits for the whole-sentence text
            self.to_rescore.append(s)
        else:
            self.sentence_audio.pop(s.id, None)
        self._cancel_stale_draft(s)
        self.pending[s.id] = s
        self.lock.notify_all()

    def _cancel_stale_draft(self, s: Sentence) -> None:
        """Caller holds the lock. The sentence just ended; if the translator is busy with a draft of an older
        text (or of text the whole-sentence pass will replace), stop it so the final starts right away."""
        job = self.mt_job
        if job and job[0] == s.id and job[1] and (s.rescoring or job[2] != s.version):
            job[3].set()

    def _run_rescores(self) -> None:
        """ASR thread: re-recognise finished multi-chunk sentences in one pass."""
        while True:
            with self.lock:
                if not self.to_rescore:
                    return
                s = self.to_rescore.pop(0)
                audio = self.sentence_audio.pop(s.id, [])
            text = asr_text_for_mt(self.asr[s.lang].transcribe(np.concatenate(audio))) if audio else ""
            with self.lock:
                if text and not is_noise(text, s.lang, s.audio_s) and text != s.text:
                    s.chunks = [text]
                    s.version += 1
                s.rescoring = False
                snap = s.snapshot()
                self.lock.notify_all()
            self.on_event("heard", snap)

    def _close_quiet_sentences(self) -> None:
        """A sentence ends at a real pause: its last chunk ended in silence and nobody has spoken since.
        (Counting only time since the last chunk would split sentences while the next chunk is being spoken.)"""
        now = time.time()
        with self.lock:
            for ch, s in list(self.open.items()):
                if self.queued[ch]:
                    continue  # chunks still waiting for ASR may continue this sentence (slow CPU)
                cv = self.vads.get(ch)
                if cv is not None:
                    # Real audio: trust the VAD. A cut at the fading end of a sentence is marked "forced",
                    # but if no speech follows, the pause is real.
                    quiet_for = now - max(s.last_chunk_at, cv.last_speech_at)
                    pause = not cv.speaking and quiet_for >= self.s.sentence_gap_s
                else:
                    pause = not s.last_forced and now - s.last_chunk_at >= self.s.sentence_gap_s
                if pause or now - s.last_chunk_at >= self.s.chunk_hard_s + 3:  # safety net
                    self._close(ch)

    # ---- translation ----
    def _next_job(self) -> Sentence | None:
        """Caller holds the lock. Oldest sentence first; drafts only when enabled."""
        for sid in sorted(self.pending):
            s = self.pending[sid]
            if s.rescoring:
                continue
            if s.needs_translation and (s.closed or self.s.drafts):
                return s
            if s.final:
                return s  # needs its final event only
        return None

    def _mt_loop(self) -> None:
        while not self.stop_flag.is_set():
            with self.lock:
                job = self._next_job()
                if job is None:
                    self.lock.wait(timeout=0.2)
                    continue
                text, version, lang, target = job.text, job.version, job.lang, job.target
                needs = job.needs_translation
                self.mt_busy = needs
                cancel = threading.Event()
                self.mt_job = (job.id, not job.closed, version, cancel)
            tr, failed = None, False
            try:
                tr = self.mt.translate(text, lang, target, cancel=cancel) if needs else None
            except TranslationCancelled:
                tr = None  # superseded by the final version of this sentence
            except Exception as e:  # noqa: BLE001 - report, then move on instead of retrying forever
                self.on_event("error", f"{type(e).__name__}: {e}")
                failed = True
            with self.lock:
                self.mt_busy = False
                self.mt_job = None
                if cancel.is_set() and tr is None:
                    continue  # no event: the final translation is next in line
                if tr is not None and version >= job.translated_version:
                    job.translation = tr.text
                    job.translated_version = version
                    job.timings["mt_ms"] = round(1000 * tr.latency_s)
                elif failed:
                    job.translated_version = version
                job.timings["lag_ms"] = round(1000 * (time.time() - job.last_chunk_at))
                job.timings["backlog"] = self.seg_q.qsize()
                if job.final:
                    self.pending.pop(job.id, None)
                snap = job.snapshot()
            self.on_event("translated", snap)
