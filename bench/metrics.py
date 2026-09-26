"""Scoring: ASR error rates, chrF, and heuristic flags for broken translations."""
import re
from collections import Counter

import sacrebleu

from transvoice.textnorm import normalize, script_ok

_META_PREFIX = re.compile(
    r"^\s*(translation|translated text|here is|here's|note|bản dịch|dịch|翻訳|訳文)\s*[:：]", re.I
)


def asr_error_rate(refs: list[str], hyps: list[str], lang: str) -> float:
    """Corpus-level CER (Japanese) or WER (others) in percent, after normalization."""
    pairs = [(normalize(r, lang), normalize(h, lang)) for r, h in zip(refs, hyps)]
    pairs = [(r, h) for r, h in pairs if r]
    if not pairs:
        return float("nan")
    import jiwer  # only needed for ASR; keeps this module importable in the COMET environment

    refs_n = [r for r, _ in pairs]
    hyps_n = [h for _, h in pairs]
    fn = jiwer.cer if lang == "ja" else jiwer.wer
    return 100.0 * fn(refs_n, hyps_n)


def corpus_chrf(refs: list[str], hyps: list[str]) -> float:
    return sacrebleu.corpus_chrf(hyps, [refs]).score


def sentence_chrf(ref: str, hyp: str) -> float:
    return sacrebleu.sentence_chrf(hyp, [ref]).score


def _has_repetition(text: str, lang: str) -> bool:
    units = list(normalize(text, lang)) if lang == "ja" else normalize(text, lang).split()
    n = 6 if lang == "ja" else 3
    grams = Counter(tuple(units[i : i + n]) for i in range(len(units) - n + 1))
    return any(c >= 3 for c in grams.values())


def translation_flags(src: str, hyp: str, ref: str, tgt_lang: str) -> list[str]:
    """Cheap checks for outputs that are clearly broken, independent of COMET."""
    h = hyp.strip()
    if not h:
        return ["empty"]
    flags = []
    if "⁇" in h or "<unk>" in h:
        flags.append("unknown_token")  # characters missing from the model vocabulary
    if not script_ok(h, tgt_lang):
        flags.append("wrong_language")
    ratio = len(normalize(h, tgt_lang)) / max(1, len(normalize(ref, tgt_lang)))
    if ratio < 0.5:
        flags.append("too_short")
    elif ratio > 2.0:
        flags.append("too_long")
    if _has_repetition(h, tgt_lang):
        flags.append("repetition")
    if _META_PREFIX.match(h) or ("\n" in h and "\n" not in src.strip()):
        flags.append("commentary")
    return flags
