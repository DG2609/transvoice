"""Saves every session on this machine: machine-readable JSONL plus a readable Markdown transcript."""
import json
import time
from dataclasses import asdict
from pathlib import Path

from .engine import Sentence

WHO = {"them": "Họ", "me": "Tôi"}


class SessionLog:
    """Only finished sentences are written; live drafts are not."""

    def __init__(self, folder: Path):
        folder.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y-%m-%d_%H%M%S")
        self.jsonl = folder / f"{stamp}.jsonl"
        self.md = folder / f"{stamp}.md"
        self.md.write_text(f"# Phiên dịch {time.strftime('%Y-%m-%d %H:%M')}\n\n", encoding="utf-8")

    def write(self, s: Sentence) -> None:
        with open(self.jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps({**asdict(s), "text": s.text}, ensure_ascii=False) + "\n")
        clock = time.strftime("%H:%M:%S", time.localtime(s.started_at))
        lines = [f"**{clock} · {WHO[s.channel]} · {s.lang.upper()}**: {s.text}"]
        if s.translation:
            lines.append(f"> {s.target.upper()}: {s.translation}")
        with open(self.md, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n\n")
