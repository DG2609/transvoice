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
def default_mt() -> Path:
    """IQ4_NL (see download.py); installs from v0.2.0 and earlier have Tencent's Q4_K_M instead."""
    fast, old = MT_DIR / "HY-MT1.5-1.8B.i1-IQ4_NL.gguf", MT_DIR / "HY-MT1.5-1.8B-Q4_K_M.gguf"
    return old if old.exists() and not fast.exists() else fast


DEFAULT_MT = default_mt()
# Re-translating a growing sentence mostly repeats the previous translation: llama.cpp's n-gram drafter
# (shared across requests) proposes those tokens and the model checks several at once. Its defaults are
# tuned for code edits (24-token match, 48+ token drafts) and almost never fire on a sentence; a 6-token
# match with drafts of up to 16 tokens cuts the final translation time by ~14% on 4 E-cores.
FAST_MT_ARGS = ("--spec-type", "ngram-mod", "--spec-ngram-mod-n-match", "6",
                "--spec-ngram-mod-n-min", "1", "--spec-ngram-mod-n-max", "16")


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
    # Quiet needed for a soft cut. Japanese has silent stretches inside words (the closure of っ, /k/, /t/:
    # 60-150 ms), and a cut there garbles the word on both sides ("電話をかけ|かけ", "出血性|性ショック"):
    # on a TV news clip 192 ms instead of 64 ms took the final text's error from 16.5% to 11.6%. English
    # keeps 64 ms: longer English chunks get false sentence periods inside (more split sentences in FLEURS).
    chunk_quiet_s: float = 0.064
    chunk_quiet_ja_s: float = 0.192
    chunk_quiet_ramp: bool = True  # accept shorter gaps as the hard limit approaches (fewer mid-word hard cuts)
    sentence_gap_s: float = 0.8  # extra pause after a chunk that closes the sentence
    # ... when the text so far ends in a word that cannot end a sentence ("the", "and", "が", "của"):
    # the speaker is hesitating, not finished.
    sentence_gap_open_s: float = 1.2
    # ... when the text ends in a Japanese sentence ending (です, ました) right at a cut: a short pause is enough.
    # Without it such a sentence waits for the next chunk (3-5 s in news), in case it goes on ("…ですが").
    # Measured from the speech level, because music under a voice keeps the VAD from ever reporting silence.
    sentence_end_pause_s: float = 0.35
    sentence_max_s: float = 12.0  # close long sentences so the final re-translation stays fast
    drafts: bool = True  # translate unfinished sentences (live feel); False = only finished sentences
    # Drafts pause while the translator was busy this much of the last 30 s. On 4 slow cores dense news keeps
    # it saturated by finals alone (lag 12-16 s either way); pausing drafts halves the wasted draft work there,
    # leaving the CPU to the other programs.
    draft_max_load: float = 0.85
    # When a sentence ends, recognise its whole audio again: chunk cuts garble words at the joins
    # ("一番一番は", "金金属"), and the final translation should not inherit that.
    rescore_final: bool = True
    # Only where it pays off: Japanese COMET 0.833 -> 0.858 and no bad translations; English WER improves but
    # the translation does not (HY-MT copes with the small errors), while the final comes 1.5 s later.
    # Vietnamese (30M-parameter zipformer) costs almost nothing to re-run.
    rescore_langs: tuple[str, ...] = ("ja", "vi")
    # Up to this long. Once only Japanese is re-recognised, 16 s covers the long run-on sentences of news and
    # interviews (FLEURS ja CER 8.2% -> 6.6%); they used to keep the garbled words of every chunk join.
    rescore_max_s: float = 16.0
    # Faster re-translation: reuse the KV cache of the shared prompt prefix and draft tokens with n-gram
    # lookup (FAST_MT_ARGS). With IQ4_NL weights the output is identical to plain decoding.
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
    last_gap: float = float("inf")  # quiet at that cut (s): a short one is often inside a word
    lang_confirmed: bool = True  # False while the language is a guess from too little audio
    shared_audio: bool = False  # split inside a chunk: its audio also holds part of a neighbouring sentence
    rescoring: bool = False  # whole-sentence recognition running; the translator waits for it
    audio_s: float = 0.0
    closed: bool = False
    dropped: bool = False  # withdrawn: in a language we do not translate
    version: int = 0  # bumps whenever a chunk is appended
    translated_version: int = 0
    translation: str | None = None
    partial: str | None = None  # translation being generated right now (streamed), shown before it completes
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


# Words a sentence cannot end with: after them the speaker is only pausing.
_OPEN_END = {
    "en": re.compile(r"\b(the|a|an|and|or|but|of|to|in|on|at|for|with|from|by|as|than|that|which|who|whose|"
                     r"because|so|if|when|while|is|are|was|were|be|been|has|have|had|will|would|can|could|"
                     r"my|your|our|their|his|her|its|this|these|those|very|not|about|into)\W*$", re.I),
    "ja": re.compile(r"(が|けど|けれど|て|で|し|から|ので|のに|と|は|を|に|へ|も|の|や|って|たり|ながら|ば|、)\W*$"),
    "vi": re.compile(r"(^|\s)(và|của|là|thì|mà|những|các|để|với|cho|nhưng|vì|nên|khi|nếu|rằng|một|trong|ở|từ|"
                     r"đến|được|bị|có|không|rất)\W*$", re.I),
}


def open_ended(text: str, lang: str) -> bool:
    rx = _OPEN_END.get(lang)
    return bool(rx and rx.search(text.strip()))


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

    def __init__(self, vad, soft_s: float, hard_s: float, quiet_s: float = 0.064, ramp: bool = True,
                 ramp_from: float = 0.0):
        self.ramp, self.ramp_from = ramp, ramp_from  # the shorter gaps are accepted from soft + ramp_from * span
        self.set_quiet(quiet_s)
        self.vad = vad
        self.soft, self.hard = int(soft_s * SAMPLE_RATE), int(hard_s * SAMPLE_RATE)
        self.pending = np.zeros(0, np.float32)
        self.run = 0  # samples of uninterrupted speech in the open segment
        self.levels: list[float] = []  # RMS of each window in the open segment
        self.quiet = 0  # consecutive quiet windows after the soft limit
        self.last_speech_at = 0.0  # wall clock of the last window the VAD considered speech
        self.cut_level = 0.0  # speech level of the segment cut last
        self.cut_quiet = -1  # quiet windows around the last cut so far; -1 once the speaker goes on

    def accept(self, samples: np.ndarray) -> list[tuple[np.ndarray, bool, float]]:
        """Returns closed segments as (samples, forced, gap): forced = cut mid-speech, the sentence goes on;
        gap = seconds of quiet at the cut (0 for a cut at the hard limit, inf for the end of speech)."""
        out = []
        buf = np.concatenate([self.pending, samples])
        n = len(buf) // self.WINDOW * self.WINDOW
        for i in range(0, n, self.WINDOW):
            w = buf[i : i + self.WINDOW]
            self.vad.accept_waveform(w)
            out += self._pop(forced=False)
            level = float(np.sqrt(np.mean(w * w)))
            if self.cut_quiet >= 0:
                self.cut_quiet = self.cut_quiet + 1 if level < 0.3 * self.cut_level else -1
            if not self.vad.is_speech_detected():
                self.run, self.levels, self.quiet = 0, [], 0
                continue
            self.last_speech_at = time.time()
            self.run += self.WINDOW
            self.levels.append(level)
            if self.run >= self.soft and level < 0.3 * float(np.median(self.levels)):
                self.quiet += 1
            else:
                self.quiet = 0
            if self.quiet >= self._needed_quiet() or self.run >= self.hard:
                self.cut_level, self.cut_quiet = float(np.median(self.levels)), self.quiet
                self.vad.flush()  # closes the segment here; detection continues with the next window
                self.run, self.levels, self.quiet = 0, [], 0
                out += self._pop(forced=True, gap=self.cut_quiet * self.WINDOW / SAMPLE_RATE)
        self.pending = buf[n:]
        return out

    MIN_QUIET_WINDOWS = 2  # 64 ms: still better than the hard cut, which lands anywhere, often mid-word

    def set_quiet(self, quiet_s: float) -> None:
        # A cut needs this many 32 ms windows well below the speech level: a gap between words, not a soft syllable.
        self.quiet_windows = max(1, round(quiet_s * SAMPLE_RATE / self.WINDOW))

    def _needed_quiet(self) -> int:
        """Right after the soft limit only a real pause cuts; the closer the hard limit, the shorter the gap that
        will do (a speaker without pauses would otherwise be cut mid-word at the hard limit)."""
        if not self.ramp or self.quiet_windows <= self.MIN_QUIET_WINDOWS:
            return self.quiet_windows
        span = self.hard - self.soft
        progress = min(1.0, max(0.0, (self.run - self.soft - self.ramp_from * span) / ((0.9 - self.ramp_from) * span)))
        return round(self.quiet_windows - progress * (self.quiet_windows - self.MIN_QUIET_WINDOWS))

    @property
    def speaking(self) -> bool:
        return self.run > 0

    @property
    def pause_after_cut_s(self) -> float:
        """How long the speaker has been quiet at the last cut (before and after it); 0 once they went on."""
        return max(0, self.cut_quiet) * self.WINDOW / SAMPLE_RATE

    def flush(self) -> list[tuple[np.ndarray, bool]]:
        self.vad.flush()
        self.run, self.levels, self.quiet = 0, [], 0
        return self._pop(forced=False)

    def _pop(self, forced: bool, gap: float = float("inf")) -> list[tuple[np.ndarray, bool, float]]:
        out = []
        while not self.vad.empty():
            out.append((np.array(self.vad.front.samples, dtype=np.float32), forced, gap))
            self.vad.pop()
        return out


class Engine:
    def __init__(self, settings: Settings, on_event: Callable[[str, object], None]):
        self.s = settings
        # Events: ("status", str) | ("heard", Sentence) | ("partial", Sentence) | ("translated", Sentence)
        # | ("error", str).
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
        # Where the translator's time goes (seconds, counts): drafts, finals, work thrown away by a cancel.
        self.mt_stats: collections.Counter = collections.Counter()
        self.mt_busy_log: collections.deque = collections.deque(maxlen=200)  # (start, end) of translations
        self.mt_job_started = 0.0
        self.threads: list[threading.Thread] = []

    # ---- lifecycle ----
    def start(self) -> None:
        self.on_event("status", "Đang tải model…")
        needed = {self.s.my_lang} | (set(self.s.langs) if self.s.their_lang == "auto" else {self.s.their_lang})
        for lang in sorted(needed):
            self.asr[lang] = ASR[self.s.asr[lang]].build(lang, self.s.asr_threads)
        if self.s.their_lang == "auto":
            self.lid = LanguageId(self.s.asr_threads)
        fast = dict(reuse_prefix=True, extra_args=FAST_MT_ARGS) if self.s.mt_fast else {}
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
                                             self.s.chunk_soft_s, self.s.chunk_hard_s, self._chunk_quiet(channel),
                                             ramp=bool(self.s.chunk_quiet_ramp))
        return self.vads[channel]

    def _chunk_quiet(self, channel: str, lang: str | None = None) -> float:
        """Quiet needed to cut a chunk, for the language being spoken on this channel."""
        if lang is None:
            lang = self.s.my_lang if channel == "me" else (
                self.s.their_lang if self.s.their_lang != "auto" else self.their_last)
        return self.s.chunk_quiet_ja_s if lang == "ja" else self.s.chunk_quiet_s

    def _vad_loop(self) -> None:
        while not self.stop_flag.is_set():
            try:
                channel, samples = self.audio_q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                if channel == "__flush__":
                    for ch, cv in self.vads.items():
                        for seg, forced, gap in cv.flush():
                            self._enqueue(ch, seg, time.time(), forced, gap)
                    self.seg_q.put(("__flush__", None, time.time(), False, float("inf")))
                    continue
                cv = self._vad_for(channel)
                for seg, forced, gap in cv.accept(self.agc.setdefault(channel, AutoGain())(samples)):
                    self._enqueue(channel, seg, time.time(), forced, gap)
            finally:
                self.audio_q.task_done()

    # ---- recognition ----
    LID_CONFIRM_S = 2.0  # language ID on at least this much audio is trusted (~94% at 3 s in the benchmark)
    # Languages Whisper confuses with the ones we handle (Japanese heard as Chinese or Korean): not foreign.
    LID_CONFUSABLE = {"zh", "yue", "ko"}

    def _foreign(self, lang: str, samples_n: int) -> bool:
        """Clearly another language (a Nepali interview in Japanese news): recognising it as Japanese gives
        nonsense that the translator turns into a fluent, invented sentence, so it is not translated at all."""
        return (lang not in self.s.langs and lang not in self.LID_CONFUSABLE
                and samples_n >= self.LID_CONFIRM_S * SAMPLE_RATE)

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
            # Trusted only if it agrees with the previous sentence: a first sentence or a switch of language is
            # re-checked on more audio (Whisper heard 3 s of Japanese news as Vietnamese; 7 s were clear).
            agrees = lang == self.their_last
            self.their_last = lang
            return lang, agrees and len(samples) >= self.LID_CONFIRM_S * SAMPLE_RATE
        # Another language on one chunk is only a suspicion (street noise can sound like anything): the sentence
        # keeps the last language, unconfirmed, and is dropped only if the check on more audio agrees.
        return self.their_last or next(l for l in self.s.langs if l != self.s.my_lang), False

    def _recheck_language(self, s: Sentence, audio: list[np.ndarray]) -> tuple[str | None, bool, list[str] | None]:
        """Re-run language ID on the whole sentence so far; if it changed, re-recognise earlier chunks.
        None = the sentence is in a language we do not translate."""
        joined = np.concatenate(audio)
        if len(joined) < self.LID_CONFIRM_S * SAMPLE_RATE:
            return s.lang, False, None
        detected = self.lid.detect(joined)
        if self._foreign(detected, len(joined)):
            return None, True, None
        if detected not in self.s.langs or detected == s.lang:
            return s.lang, True, None
        self.their_last = detected
        texts = [asr_text_for_mt(self.asr[detected].transcribe(a)) for a in audio[:-1]]
        return detected, True, [t for t, a in zip(texts, audio) if not is_noise(t, detected, len(a) / SAMPLE_RATE)]

    def _target(self, channel: str) -> str:
        if channel == "them":
            return self.s.my_lang
        return self.their_last or next(l for l in self.s.langs if l != self.s.my_lang)

    def _enqueue(self, channel: str, samples: np.ndarray, ended_at: float, forced: bool,
                 gap: float = float("inf")) -> None:
        with self.lock:
            self.queued[channel] += 1
        self.seg_q.put((channel, samples, ended_at, forced, gap))

    def _asr_loop(self) -> None:
        while not self.stop_flag.is_set():
            try:
                channel, samples, ended_at, forced, gap = self.seg_q.get(timeout=0.2)
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
                            self._close(ch, "flush")
                    self._run_rescores()
                    continue
                self._add_chunk(channel, samples, ended_at, forced, gap)
            except Exception as e:  # noqa: BLE001 - keep the app alive, report the failure
                self.on_event("error", f"{type(e).__name__}: {e}")
            finally:
                self.seg_q.task_done()
            self._close_quiet_sentences()
            self._run_rescores()

    def _add_chunk(self, channel: str, samples: np.ndarray, ended_at: float, forced: bool,
                   gap: float = float("inf")) -> None:
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
        if lang is None:
            if current is not None:
                self._drop(current)
            return
        t1 = time.perf_counter()
        text = asr_text_for_mt(self.asr[lang].transcribe(samples))
        t2 = time.perf_counter()
        if forced:
            # The ASR models end every chunk with a period, even one cut mid-sentence ("On August. 15",
            # "けがをしていて警察は。"); a false period would split the sentence at every cut. Japanese
            # sentence endings are words (です, ました), so those still end a sentence without the period.
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
                        and not continues_sentence(piece, lang) and self._trusted_end(s):
                    # The previous chunk ended a sentence exactly at the cut ("…思いますか" | "やっぱ…").
                    self._close(channel, "end")
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
                s.last_forced, s.last_gap = forced, gap
                s.timings.update(lid_ms=round(1000 * (t1 - t0)), asr_ms=round(1000 * (t2 - t1)))
                self.pending[s.id] = s
                last = i == len(pieces) - 1
                too_long = (s.audio_s >= self.s.sentence_max_s and not forced) \
                    or s.audio_s >= 1.25 * self.s.sentence_max_s  # prefer closing at a pause, not mid-word
                if not last or (not forced and ends_sentence(piece, lang)) or too_long:
                    self._close(channel, "split" if not last else "max" if too_long else "end")
                snaps.append(s.snapshot())
                s = None if not last else s
            if not pieces:  # relabel only
                self.pending[s.id] = s
                snaps.append(s.snapshot())
            self.lock.notify_all()
        cv = self.vads.get(channel)
        if cv is not None:  # the next cut follows the language now being spoken
            cv.set_quiet(self._chunk_quiet(channel, lang))
        for snap in snaps:
            self.on_event("heard", snap)

    def _rescore(self, s: Sentence, audio: list[np.ndarray]) -> None:
        """ASR thread, no lock held: replace the chunk-by-chunk text with one pass over the whole audio."""
        t0 = time.perf_counter()
        text = asr_text_for_mt(self.asr[s.lang].transcribe(np.concatenate(audio)))
        with self.lock:
            changed = bool(text) and not is_noise(text, s.lang, s.audio_s) and text != s.text
            if changed:
                s.chunks = [text]
                s.version += 1
            s.rescoring = False
            s.timings.update(rescore_ms=round(1000 * (time.perf_counter() - t0)), rescore_changed=changed)
            self._cancel_stale_draft(s)
            snap = s.snapshot()
            self.lock.notify_all()
        self.on_event("heard", snap)

    def _trusted_end(self, s: Sentence) -> bool:
        """Does the ending of the last chunk really end the sentence? At the end of speech, yes. At a forced cut,
        Latin punctuation is only an ASR guess; Japanese endings are words, but only a cut in a real pause
        is trusted: a cut in the silence inside a word makes the ASR invent one ("作っ|たり" -> "作っています")."""
        if not s.last_forced:
            return True
        return s.lang == "ja" and s.last_gap >= self.s.chunk_quiet_ja_s - 1e-6

    def _drop(self, s: Sentence) -> None:
        """Withdraw an open sentence that turned out to be in another language (its draft line disappears)."""
        with self.lock:
            if self.open.get(s.channel) is s:
                del self.open[s.channel]
            self.pending.pop(s.id, None)
            self.sentence_audio.pop(s.id, None)
            job = self.mt_job
            if job and job[0] == s.id:
                job[3].set()
            s.dropped = True
            snap = s.snapshot()
        self.on_event("dropped", snap)

    def _new_sentence(self, channel: str, lang: str, started_at: float) -> Sentence:
        """Caller holds the lock."""
        s = Sentence(next(self.ids), channel, lang, self._target(channel), started_at=started_at)
        self.open[channel] = s
        return s

    def _close(self, channel: str, reason: str) -> None:
        """Caller holds the lock. `reason`: end (sentence ending), pause, split, max (too long), flush..."""
        s = self.open.pop(channel, None)
        if s is None:
            return
        s.closed = True
        s.timings["close_ms"] = round(1000 * (time.time() - s.last_chunk_at))  # pause detection
        s.timings["close"] = reason
        audio = self.sentence_audio.get(s.id, [])
        if (self.s.rescore_final and s.lang in self.s.rescore_langs and not s.shared_audio
                and len(s.chunks) >= 2 and len(audio) >= 2
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
        """Caller holds the lock. `s` just ended, or its whole-sentence text is ready: its final translation
        goes first. Stop a draft of an older text of `s` (or of text the whole-sentence pass will replace),
        and, once the final can start, a draft of a later sentence: that one is redone afterwards with more
        text anyway, while the final would otherwise wait seconds behind it (real speech: the next sentence
        starts while this one is being re-recognised)."""
        job = self.mt_job
        if not job or not job[1]:
            return  # idle, or already translating a final
        if job[0] == s.id:
            stale = s.rescoring or job[2] != s.version
        else:
            stale = s.closed and not s.rescoring and s.needs_translation and job[0] > s.id
        if stale:
            job[3].set()

    def _run_rescores(self) -> None:
        """ASR thread: re-recognise finished multi-chunk sentences in one pass."""
        while True:
            with self.lock:
                if not self.to_rescore:
                    return
                s = self.to_rescore.pop(0)
                audio = self.sentence_audio.pop(s.id, [])
            if audio:
                self._rescore(s, audio)
            else:
                with self.lock:
                    s.rescoring = False
                    self._cancel_stale_draft(s)
                    self.lock.notify_all()

    def _close_quiet_sentences(self) -> None:
        """A sentence ends at a real pause: its last chunk ended in silence and nobody has spoken since.
        (Counting only time since the last chunk would split sentences while the next chunk is being spoken.)"""
        now = time.time()
        with self.lock:
            for ch, s in list(self.open.items()):
                if self.queued[ch]:
                    continue  # chunks still waiting for ASR may continue this sentence (slow CPU)
                cv = self.vads.get(ch)
                gap = self.s.sentence_gap_open_s if open_ended(s.text, s.lang) else self.s.sentence_gap_s
                ended = False
                if cv is not None:
                    # Real audio: trust the VAD. A cut at the fading end of a sentence is marked "forced",
                    # but if no speech follows, the pause is real.
                    quiet_for = now - max(s.last_chunk_at, cv.last_speech_at)
                    pause = not cv.speaking and quiet_for >= gap
                    ended = (s.last_forced and s.lang == "ja" and ends_sentence(s.chunks[-1], s.lang)
                             and cv.pause_after_cut_s >= self.s.sentence_end_pause_s)
                else:
                    pause = not s.last_forced and now - s.last_chunk_at >= gap
                if ended:
                    self._close(ch, "end")
                elif pause or now - s.last_chunk_at >= self.s.chunk_hard_s + 3:  # safety net
                    self._close(ch, "pause" if pause else "timeout")

    # ---- translation ----
    def _mt_load(self, window: float = 30.0) -> float:
        """Caller holds the lock. Share of the last `window` seconds the translator was busy."""
        now = time.time()
        busy = sum(max(0.0, min(end, now) - max(start, now - window)) for start, end in self.mt_busy_log)
        if self.mt_job is not None:
            busy += now - max(self.mt_job_started, now - window)
        return busy / window

    def _next_job(self) -> Sentence | None:
        """Caller holds the lock. Oldest sentence first; drafts only when enabled and the translator keeps up."""
        drafts = self.s.drafts and self._mt_load() < self.s.draft_max_load
        for sid in sorted(self.pending):
            s = self.pending[sid]
            if s.rescoring:
                continue
            if s.needs_translation and (s.closed or drafts):
                return s
            if s.final:
                return s  # needs its final event only
        return None

    def _partial_sink(self, job: Sentence) -> Callable[[str], None]:
        """Streamed words of a translation -> "partial" events, at most every 100 ms. A partial never
        replaces a longer draft that is already on screen (no shrinking subtitles)."""
        last = [0.0]

        def sink(text: str) -> None:
            now = time.time()
            if now - last[0] < 0.1:
                return
            with self.lock:
                if job.dropped or (job.translation and len(text) < len(job.translation)):
                    return
                job.partial = text
                if job.closed and "final_first_ms" not in job.timings:
                    job.timings["final_first_ms"] = round(1000 * (now - job.last_chunk_at))
                snap = job.snapshot()
            last[0] = now
            self.on_event("partial", snap)

        return sink

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
                if job.closed and needs:  # final translation: how long it waited after the close
                    job.timings["final_start_ms"] = round(1000 * (time.time() - job.last_chunk_at))
                self.mt_job = (job.id, not job.closed, version, cancel)
                self.mt_job_started = time.time()
            tr, failed = None, False
            t0 = time.perf_counter()
            try:
                tr = self.mt.translate(text, lang, target, cancel=cancel,
                                       on_partial=self._partial_sink(job)) if needs else None
            except TranslationCancelled:
                tr = None  # superseded by the final version of this sentence
            except Exception as e:  # noqa: BLE001 - report, then move on instead of retrying forever
                self.on_event("error", f"{type(e).__name__}: {e}")
                failed = True
            with self.lock:
                self.mt_busy = False
                self.mt_job = None
                job.partial = None
                if needs:
                    self.mt_busy_log.append((self.mt_job_started, time.time()))
                    kind = "cancelled" if cancel.is_set() and tr is None else "draft" if not job.closed else "final"
                    self.mt_stats[kind] += 1
                    self.mt_stats[kind + "_s"] += time.perf_counter() - t0
                if (cancel.is_set() and tr is None) or job.dropped:
                    continue  # no event: the final translation is next in line, or the sentence was withdrawn
                if tr is not None and version >= job.translated_version:
                    job.translation = tr.text
                    job.translated_version = version
                    job.timings["mt_ms"] = round(1000 * tr.latency_s)
                    if job.closed:
                        job.timings["final_mt_ms"] = round(1000 * tr.latency_s)
                elif failed:
                    job.translated_version = version
                job.timings["lag_ms"] = round(1000 * (time.time() - job.last_chunk_at))
                job.timings["backlog"] = self.seg_q.qsize()
                if job.final:
                    self.pending.pop(job.id, None)
                snap = job.snapshot()
            self.on_event("translated", snap)
