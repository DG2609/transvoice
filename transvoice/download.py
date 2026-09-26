"""Model downloads (stdlib + curl.exe, which ships with Windows 10/11).

The app needs only APP_MODELS (~2.9 GB). The rest of MANIFEST is for the benchmark in scripts/.
"""
import json
import subprocess
import tarfile
import urllib.request
import zipfile
from pathlib import Path

from .osutil import llama_asset
from .paths import MODELS

SHERPA = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"
HF = "https://huggingface.co"

# kind: "tar" = sherpa-onnx tar.bz2, "files" = individual files, "hf_repo" = whole HF repo, "archive" = zip or tar.gz
MANIFEST = {
    # ---- ASR ----
    # The original multilingual SenseVoice-Small. (The newer 2025-09-09 package is a Cantonese fine-tune.)
    "sensevoice": {"kind": "files", "dir": "asr/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17",
                   "base": f"{HF}/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/resolve/main",
                   "files": ["model.int8.onnx", "tokens.txt"]},
    "zipformer-vi-30m": {"kind": "tar", "dir": "asr",
                         "url": f"{SHERPA}/sherpa-onnx-zipformer-vi-30M-int8-2026-02-09.tar.bz2"},
    # Hugging Face mirror of the sherpa-onnx release (GitHub release downloads can be throttled).
    "vietasr": {"kind": "hf_repo", "dir": "asr/sherpa-onnx-zipformer-vi-int8-2025-04-20",
                "repo": "csukuangfj/sherpa-onnx-zipformer-vi-int8-2025-04-20"},
    "nemotron": {"kind": "tar", "dir": "asr",
                 "url": f"{SHERPA}/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11.tar.bz2"},
    "reazonspeech-k2": {"kind": "files", "dir": "asr/reazonspeech-k2-v2",
                        "base": f"{HF}/reazon-research/reazonspeech-k2-v2/resolve/main",
                        "files": ["encoder-epoch-99-avg-1.int8.onnx", "decoder-epoch-99-avg-1.onnx",
                                  "joiner-epoch-99-avg-1.int8.onnx", "tokens.txt"]},
    "qwen3-asr": {"kind": "tar", "dir": "asr",
                  "url": f"{SHERPA}/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25.tar.bz2"},
    "parakeet-ja": {"kind": "hf_repo", "dir": "asr/sherpa-onnx-nemo-parakeet-tdt_ctc-0.6b-ja-35000-int8",
                    "repo": "csukuangfj/sherpa-onnx-nemo-parakeet-tdt_ctc-0.6b-ja-35000-int8"},
    "moonshine-ja": {"kind": "tar", "dir": "asr",
                     "url": f"{SHERPA}/sherpa-onnx-moonshine-base-ja-quantized-2026-02-27.tar.bz2"},
    "moonshine-vi": {"kind": "tar", "dir": "asr",
                     "url": f"{SHERPA}/sherpa-onnx-moonshine-base-vi-quantized-2026-02-27.tar.bz2"},
    "silero-vad": {"kind": "files", "dir": "vad", "base": SHERPA, "files": ["silero_vad.onnx"]},
    # Spoken language identification (Whisper encoder/decoder used only for LID)
    "whisper-tiny-lid": {"kind": "files", "dir": "lid/whisper-tiny",
                         "base": f"{HF}/csukuangfj/sherpa-onnx-whisper-tiny/resolve/main",
                         "files": ["tiny-encoder.int8.onnx", "tiny-decoder.int8.onnx"]},
    "whisper-base-lid": {"kind": "files", "dir": "lid/whisper-base",
                         "base": f"{HF}/csukuangfj/sherpa-onnx-whisper-base/resolve/main",
                         "files": ["base-encoder.int8.onnx", "base-decoder.int8.onnx"]},
    # ---- MT ----
    "shisa": {"kind": "files", "dir": "mt",
              "base": f"{HF}/mradermacher/shisa-v2.1-lfm2-1.2b-GGUF/resolve/main",
              "files": ["shisa-v2.1-lfm2-1.2b.Q4_K_M.gguf"]},
    "nllb-600m": {"kind": "hf_repo", "dir": "mt/nllb-200-distilled-600M-ct2-int8",
                  "repo": "JustFrederik/nllb-200-distilled-600M-ct2-int8"},
    "nllb-1.3b": {"kind": "hf_repo", "dir": "mt/nllb-200-distilled-1.3B-ct2-int8",
                  "repo": "JustFrederik/nllb-200-distilled-1.3B-ct2-int8"},
    "hy-mt": {"kind": "files", "dir": "mt",
              "base": f"{HF}/tencent/HY-MT1.5-1.8B-GGUF/resolve/main",
              "files": ["HY-MT1.5-1.8B-Q4_K_M.gguf"]},
    "hy-mt-q8": {"kind": "files", "dir": "mt",
                 "base": f"{HF}/tencent/HY-MT1.5-1.8B-GGUF/resolve/main",
                 "files": ["HY-MT1.5-1.8B-Q8_0.gguf"]},
    "translategemma": {"kind": "files", "dir": "mt",
                       "base": f"{HF}/mradermacher/translategemma-4b-it-GGUF/resolve/main",
                       "files": ["translategemma-4b-it.Q4_K_M.gguf"]},
    "cat-translate": {"kind": "files", "dir": "mt",
                      "base": f"{HF}/mradermacher/CAT-Translate-1.4b-GGUF/resolve/main",
                      "files": ["CAT-Translate-1.4b.Q4_K_M.gguf"]},
    "lfm2-enjp": {"kind": "files", "dir": "mt",
                  "base": f"{HF}/LiquidAI/LFM2-350M-ENJP-MT-GGUF/resolve/main",
                  "files": ["LFM2-350M-ENJP-MT-Q4_K_M.gguf"]},
    # General small LLMs: could translate with custom context AND do summary / Q&A in one model.
    "qwen3.5-2b": {"kind": "files", "dir": "mt",
                   "base": f"{HF}/unsloth/Qwen3.5-2B-GGUF/resolve/main",
                   "files": ["Qwen3.5-2B-Q4_K_M.gguf"]},
    "qwen3.5-4b": {"kind": "files", "dir": "mt",
                   "base": f"{HF}/unsloth/Qwen3.5-4B-GGUF/resolve/main",
                   "files": ["Qwen3.5-4B-Q4_K_M.gguf"]},
    # ---- evaluation ----
    "comet": {"kind": "files", "dir": "eval/wmt22-comet-da",
              "base": f"{HF}/Unbabel/wmt22-comet-da/resolve/main",
              "files": ["hparams.yaml", "checkpoints/model.ckpt"]},
    # ---- runtime ----
    # CPU build for the current OS (zip on Windows, tar.gz elsewhere; tar keeps the executable bit).
    "llama-cpp": {"kind": "archive", "dir": "bin/llama.cpp",
                  "url": f"https://github.com/ggml-org/llama.cpp/releases/download/b11179/{llama_asset()}"},
}


def curl(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    subprocess.run(["curl", "-sSfL", "--retry", "5", "-C", "-", "-o", str(part), url], check=True)
    part.replace(dest)


def hf_repo_files(repo: str) -> list[str]:
    with urllib.request.urlopen(f"{HF}/api/models/{repo}/tree/main") as r:
        return [e["path"] for e in json.load(r) if e["type"] == "file" and e["path"] != ".gitattributes"]


def fetch(name: str, spec: dict) -> None:
    target = MODELS / spec["dir"]
    kind = spec["kind"]
    if kind == "tar":
        archive = MODELS / "_downloads" / spec["url"].rsplit("/", 1)[1]
        out_dir = target / archive.name.removesuffix(".tar.bz2")
        if out_dir.exists():
            print(f"[skip] {name}")
            return
        curl(spec["url"], archive)
        with tarfile.open(archive, "r:bz2") as t:
            t.extractall(target, filter="data")
        archive.unlink()
    elif kind == "archive":
        if target.exists():
            print(f"[skip] {name}")
            return
        archive = MODELS / "_downloads" / spec["url"].rsplit("/", 1)[1]
        curl(spec["url"], archive)
        if archive.suffix == ".zip":
            with zipfile.ZipFile(archive) as z:
                z.extractall(target)
        else:
            with tarfile.open(archive, "r:gz") as t:
                t.extractall(target, filter="tar")  # "tar" keeps the executable bit and relative symlinks
        archive.unlink()
    else:
        files = spec["files"] if kind == "files" else hf_repo_files(spec["repo"])
        base = spec["base"] if kind == "files" else f"{HF}/{spec['repo']}/resolve/main"
        for f in files:
            dest = target / f
            if dest.exists():
                continue
            curl(f"{base}/{f}", dest)
    print(f"[ok] {name}", flush=True)

# What TransVoice itself uses (see docs/DECISIONS.md).
APP_MODELS = ["sensevoice", "parakeet-ja", "zipformer-vi-30m", "whisper-base-lid", "silero-vad", "hy-mt", "llama-cpp"]


def download(names: list[str]) -> None:
    for n in names:
        print(f"[...] {n}", flush=True)
        fetch(n, MANIFEST[n])
