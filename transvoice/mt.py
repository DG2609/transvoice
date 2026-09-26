"""Translation candidates: NLLB via CTranslate2 (in-process) and GGUF LLMs via llama-server."""
import re
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import requests

from .osutil import IS_WIN, child_process_kwargs, kill_with_parent
from .paths import ROOT

MT_DIR = ROOT / "models" / "mt"
LANG_NAME = {"ja": "Japanese", "en": "English", "vi": "Vietnamese"}
NLLB_CODE = {"ja": "jpn_Jpan", "en": "eng_Latn", "vi": "vie_Latn"}
ALL_PAIRS = [(s, t) for s in LANG_NAME for t in LANG_NAME if s != t]
JA_EN_PAIRS = [("ja", "en"), ("en", "ja")]


# Vietnamese weekdays are "thứ + ordinal" ("thứ năm" = Thursday, literally "fifth"), so MT models
# translate them as "the fifth day" or even the wrong weekday. Detect them and pass explicit terms.
_VI_WEEKDAYS = {
    "hai": ("Monday", "月曜日"), "2": ("Monday", "月曜日"), "ba": ("Tuesday", "火曜日"), "3": ("Tuesday", "火曜日"),
    "tư": ("Wednesday", "水曜日"), "4": ("Wednesday", "水曜日"), "năm": ("Thursday", "木曜日"), "5": ("Thursday", "木曜日"),
    "sáu": ("Friday", "金曜日"), "6": ("Friday", "金曜日"), "bảy": ("Saturday", "土曜日"), "7": ("Saturday", "土曜日"),
}
_VI_WEEKDAY_RE = re.compile(r"\b(thứ\s+(hai|ba|tư|năm|sáu|bảy|[2-7])|chủ\s+nhật)\b", re.IGNORECASE)
# "thứ hai" is also the ordinal "second" ("lần thứ hai"), so require a time-like neighbour.
_VI_BEFORE = {"vào", "sang", "ngày", "tối", "sáng", "chiều", "trưa", "đêm", "hôm", "đến", "tới", "từ", "hẹn", "các", "mỗi", "và", "hoặc"}
_VI_AFTER = {"này", "tới", "tuần", "sau", "trước", "hàng", "và", "hoặc", "là", "thì", "được", "nhé", "nha", "anh", "chị", "em"}


def glossary_terms(text: str, src: str, tgt: str) -> list[tuple[str, str]]:
    if src != "vi" or tgt not in ("en", "ja"):
        return []
    terms = []
    for m in _VI_WEEKDAY_RE.finditer(text):
        before = text[: m.start()].split()
        after = text[m.end():].split()
        prev = before[-1].lower().strip(",.") if before else ""
        nxt = after[0].lower().strip(",.!?") if after else ""
        if m.group(0).lower().startswith("chủ"):
            en, ja = "Sunday", "日曜日"
        elif text[m.end(): m.end() + 1] == ",":
            continue  # "Thứ hai, ..." = "Secondly, ..."
        elif not before or prev in _VI_BEFORE or nxt in _VI_AFTER or not after:
            en, ja = _VI_WEEKDAYS[m.group(2).lower()]
        else:
            continue
        terms.append((m.group(0), en if tgt == "en" else ja))
    return terms


@dataclass
class Translation:
    text: str
    latency_s: float
    out_tokens: int | None = None


class NllbEngine:
    def __init__(self, model_dir: Path, threads: int):
        import ctranslate2
        import sentencepiece as spm

        self.sp = spm.SentencePieceProcessor(model_file=str(model_dir / "sentencepiece.bpe.model"))
        self.tr = ctranslate2.Translator(
            str(model_dir), device="cpu", compute_type="int8", intra_threads=threads, inter_threads=1
        )
        self.pids: list[int] = []  # runs in-process

    def translate(self, text: str, src: str, tgt: str) -> Translation:
        t0 = time.perf_counter()
        tokens = [NLLB_CODE[src], *self.sp.encode(text, out_type=str), "</s>"]
        res = self.tr.translate_batch(
            [tokens], target_prefix=[[NLLB_CODE[tgt]]], beam_size=4, max_decoding_length=256
        )
        out = res[0].hypotheses[0][1:]
        return Translation(self.sp.decode(out), time.perf_counter() - t0, len(out))

    def close(self) -> None:
        pass


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def llama_server_exe() -> Path:
    name = "llama-server.exe" if IS_WIN else "llama-server"
    found = list((ROOT / "models" / "bin").rglob(name))
    if not found:
        raise FileNotFoundError(f"{name} not found under models/bin (run --download-models)")
    return found[0]


# A prompt builder returns either {"messages": [...]} (chat endpoint, GGUF template applied by the
# server) or {"prompt": str, "stop": [...]} (raw completion endpoint).
PromptFn = Callable[[str, str, str], dict]


class LlamaEngine:
    def __init__(self, gguf: Path, prompt: PromptFn, threads: int, ctx: int = 2048, jinja: bool = True,
                 low_priority: bool = False):
        self.prompt = prompt
        self.port = _free_port()
        self.proc = subprocess.Popen(
            [str(llama_server_exe()), "-m", str(gguf), "-t", str(threads), "-c", str(ctx),
             "-np", "1", "--host", "127.0.0.1", "--port", str(self.port), "--jinja" if jinja else "--no-jinja",
             # Default host-RAM prompt cache (8 GiB) and context checkpoints make RSS grow with every
             # request; each translation is independent, so keep memory flat.
             "--cache-ram", "0", "--ctx-checkpoints", "0"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **child_process_kwargs(low_priority),
        )
        kill_with_parent(self.proc)
        self.pids = [self.proc.pid]
        self.url = f"http://127.0.0.1:{self.port}"
        deadline = time.time() + 180
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited with {self.proc.returncode} for {gguf.name}")
            try:
                if requests.get(f"{self.url}/health", timeout=1).status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(0.5)
        raise TimeoutError("llama-server did not become healthy")

    def translate(self, text: str, src: str, tgt: str) -> Translation:
        req = self.prompt(text, src, tgt)
        common = {"temperature": 0, "repeat_penalty": 1.05, "cache_prompt": False}
        t0 = time.perf_counter()
        if "messages" in req:
            r = requests.post(f"{self.url}/v1/chat/completions",
                              json={"messages": req["messages"], "max_tokens": 384, "stop": req.get("stop", []),
                                    **common, **req.get("extra", {})}, timeout=300)
            r.raise_for_status()
            body = r.json()
            out = body["choices"][0]["message"]["content"]
            n = body.get("usage", {}).get("completion_tokens")
        else:
            r = requests.post(f"{self.url}/completion",
                              json={"prompt": req["prompt"], "stop": req["stop"], "n_predict": 384, **common},
                              timeout=300)
            r.raise_for_status()
            body = r.json()
            out = body["content"]
            n = body.get("tokens_predicted")
        out = out.strip()
        if "\n\n" not in text:
            # Some chat models append "(Note: ...)" after a blank line; an app would keep only the first block.
            out = out.split("\n\n", 1)[0].strip()
        return Translation(out, time.perf_counter() - t0, n)

    def close(self) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()


# ---- prompt formats, taken from each model card ----

def _hy_mt(text, src, tgt):
    return {"messages": [{"role": "user", "content":
            f"Translate the following segment into {LANG_NAME[tgt]}, without additional explanation.\n\n{text}"}]}


def user_glossary_terms(text: str, src: str, tgt: str, entries) -> list[tuple[str, str]]:
    """Entries are dicts like {"vi": "Công ty ABC", "ja": "ABC社", "en": "ABC Corp."}."""
    low = text.lower()
    return [(e[src], e[tgt]) for e in entries if src in e and tgt in e and e[src].lower() in low]


def make_hy_mt_prompt(user_entries=()) -> PromptFn:
    """HY-MT's terminology-intervention format, used only when the source contains known terms."""

    def prompt(text, src, tgt):
        terms = glossary_terms(text, src, tgt) + user_glossary_terms(text, src, tgt, user_entries)
        if not terms:
            return _hy_mt(text, src, tgt)
        ref = "\n".join(f"{s} is translated as {t}" for s, t in terms)
        return {"messages": [{"role": "user", "content":
                f"Refer to the following translations:\n{ref}\n\n"
                f"Translate the following segment into {LANG_NAME[tgt]}, without additional explanation.\n\n{text}"}]}

    return prompt


_hy_mt_glossary = make_hy_mt_prompt()


def _translategemma(text, src, tgt):
    s, t = LANG_NAME[src], LANG_NAME[tgt]
    prompt = (
        f"<start_of_turn>user\nYou are a professional {s} ({src}) to {t} ({tgt}) translator. "
        f"Your goal is to accurately convey the meaning and nuances of the original {s} text while "
        f"adhering to {t} grammar, vocabulary, and cultural sensitivities.\n"
        f"Produce only the {t} translation, without any additional explanations or commentary. "
        f"Please translate the following {s} text into {t}:\n\n\n{text}<end_of_turn>\n<start_of_turn>model\n"
    )
    return {"prompt": prompt, "stop": ["<end_of_turn>"]}


def _cat_translate(text, src, tgt):
    # The GGUF does not mark </s> as end-of-generation, so it would be emitted as text.
    return {"messages": [{"role": "user", "content":
            f"Translate the following {LANG_NAME[src]} text into {LANG_NAME[tgt]}.\n\n{text}"}],
            "stop": ["</s>"]}


def _lfm2_enjp(text, src, tgt):
    system = "Translate to Japanese." if tgt == "ja" else "Translate to English."
    return {"messages": [{"role": "system", "content": system}, {"role": "user", "content": text}]}


def _shisa(text, src, tgt):
    system = (f"You are a professional translator. Translate the user's {LANG_NAME[src]} text into "
              f"{LANG_NAME[tgt]}. Output only the translated text: no notes, explanations or comments.")
    return {"messages": [{"role": "system", "content": system}, {"role": "user", "content": text}]}


def _general_llm(text, src, tgt):
    # Same instruction as shisa; thinking mode off so the model answers directly.
    return {**_shisa(text, src, tgt), "extra": {"chat_template_kwargs": {"enable_thinking": False}}}


@dataclass
class MtSpec:
    pairs: list[tuple[str, str]]
    build: Callable[[int], object]  # threads -> engine
    note: str = ""


def _gguf(name: str) -> Path:
    return MT_DIR / name


REGISTRY: dict[str, MtSpec] = {
    "nllb-600m": MtSpec(ALL_PAIRS, lambda t: NllbEngine(MT_DIR / "nllb-200-distilled-600M-ct2-int8", t), "CT2 int8, beam 4"),
    "nllb-1.3b": MtSpec(ALL_PAIRS, lambda t: NllbEngine(MT_DIR / "nllb-200-distilled-1.3B-ct2-int8", t), "CT2 int8, beam 4"),
    "hy-mt-1.8b": MtSpec(ALL_PAIRS, lambda t: LlamaEngine(_gguf("HY-MT1.5-1.8B-Q4_K_M.gguf"), _hy_mt, t), "Q4_K_M"),
    "hy-mt-1.8b-q8": MtSpec(ALL_PAIRS, lambda t: LlamaEngine(_gguf("HY-MT1.5-1.8B-Q8_0.gguf"), _hy_mt, t), "Q8_0"),
    "hy-mt-1.8b-gloss": MtSpec(ALL_PAIRS, lambda t: LlamaEngine(_gguf("HY-MT1.5-1.8B-Q4_K_M.gguf"), _hy_mt_glossary, t), "Q4_K_M + auto glossary"),
    # Its Jinja chat template rejects plain-string content, so skip template parsing and use the raw prompt.
    "translategemma-4b": MtSpec(ALL_PAIRS, lambda t: LlamaEngine(_gguf("translategemma-4b-it.Q4_K_M.gguf"), _translategemma, t, jinja=False), "Q4_K_M"),
    "cat-translate-1.4b": MtSpec(JA_EN_PAIRS, lambda t: LlamaEngine(_gguf("CAT-Translate-1.4b.Q4_K_M.gguf"), _cat_translate, t), "Q4_K_M, JA/EN only"),
    "lfm2-350m-enjp": MtSpec(JA_EN_PAIRS, lambda t: LlamaEngine(_gguf("LFM2-350M-ENJP-MT-Q4_K_M.gguf"), _lfm2_enjp, t), "Q4_K_M, JA/EN only"),
    "shisa-1.2b": MtSpec(JA_EN_PAIRS, lambda t: LlamaEngine(_gguf("shisa-v2.1-lfm2-1.2b.Q4_K_M.gguf"), _shisa, t), "Q4_K_M, JA/EN only"),
    "qwen3.5-2b": MtSpec(ALL_PAIRS, lambda t: LlamaEngine(_gguf("Qwen3.5-2B-Q4_K_M.gguf"), _general_llm, t), "Q4_K_M, general LLM"),
    "qwen3.5-4b": MtSpec(ALL_PAIRS, lambda t: LlamaEngine(_gguf("Qwen3.5-4B-Q4_K_M.gguf"), _general_llm, t), "Q4_K_M, general LLM"),
}
