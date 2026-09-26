"""Benchmark ASR models on CPU: accuracy, speed, CPU and RAM.

Each model runs in its own subprocess so RAM numbers are not polluted by other models.

  python scripts/run_asr.py                       # all models, all datasets
  python scripts/run_asr.py --models sensevoice,nemotron --n 50
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import psutil  # noqa: E402

from transvoice.asr import REGISTRY  # noqa: E402
from bench.data import ROOT, asr_dataset  # noqa: E402
from bench.metrics import asr_error_rate  # noqa: E402
from bench.resources import ResourceMonitor, limit_cpus  # noqa: E402

OUT = ROOT / "results" / "asr"


def jobs(models: list[str], datasets: list[str], langs: list[str]) -> list[tuple[str, str, str]]:
    out = []
    for m in models:
        for ds in datasets:
            for lang in REGISTRY[m].langs:
                if lang in langs and not (ds == "reazon" and lang != "ja"):
                    out.append((m, ds, lang))
    return out


def run_one(model: str, ds: str, lang: str, n: int, threads: int, speed_only: bool = False) -> dict:
    utts = asr_dataset(ds, lang, n)
    audio = [u.load() for u in utts]
    me = psutil.Process(os.getpid())
    base_rss = me.memory_info().rss / 2**20

    t0 = time.perf_counter()
    asr = REGISTRY[model].build(lang, threads)
    load_s = time.perf_counter() - t0
    asr.transcribe(audio[0][: 16000 * 2])  # warm-up
    if hasattr(asr, "chunk_ms"):
        asr.chunk_ms.clear()

    rows, proc = [], []
    with ResourceMonitor([os.getpid()]) as mon:
        for u, a in zip(utts, audio):
            t = time.perf_counter()
            hyp = asr.transcribe(a)
            dt = time.perf_counter() - t
            proc.append(dt)
            rows.append({"key": u.key, "ref": u.text, "hyp": hyp, "dur": round(u.duration, 3), "proc_s": round(dt, 4)})

    OUT.mkdir(parents=True, exist_ok=True)
    if not speed_only:
        with open(OUT / f"{model}__{ds}-{lang}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    audio_s = sum(u.duration for u in utts)
    summary = {
        "model": model,
        "dataset": ds,
        "lang": lang,
        "n": len(rows),
        "metric": "CER" if lang == "ja" else "WER",
        "error": round(asr_error_rate([r["ref"] for r in rows], [r["hyp"] for r in rows], lang), 2),
        "rtf": round(sum(proc) / audio_s, 4),
        "p50_ms": round(1000 * statistics.median(proc)),
        "p95_ms": round(1000 * sorted(proc)[int(0.95 * (len(proc) - 1))]),
        "avg_cores": round(mon.avg_cores, 2),
        "model_ram_mb": round(mon.peak_rss_mb - base_rss),
        "load_s": round(load_s, 2),
        "threads": threads,
    }
    if getattr(asr, "chunk_ms", None):
        ch = sorted(asr.chunk_ms)
        summary["chunk_p50_ms"] = round(statistics.median(ch), 1)
        summary["chunk_p95_ms"] = round(ch[int(0.95 * (len(ch) - 1))], 1)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(REGISTRY))
    ap.add_argument("--datasets", default="fleurs,reazon")
    ap.add_argument("--langs", default="ja,en,vi")
    ap.add_argument("--n", type=int, default=150, help="utterances per FLEURS language")
    ap.add_argument("--n-reazon", type=int, default=300)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--speed-only", action="store_true",
                    help="timing pass on an idle machine: no per-utterance files, results go to speed.jsonl")
    ap.add_argument("--cpus", help="pin to these logical CPUs, e.g. 12,13,14,15 (E-cores = office-PC profile)")
    ap.add_argument("--one", nargs=3, metavar=("MODEL", "DATASET", "LANG"), help=argparse.SUPPRESS)
    args = ap.parse_args()
    limit_cpus(args.cpus)  # child processes inherit the affinity

    if args.one:
        m, ds, lang = args.one
        n = args.n_reazon if ds == "reazon" else args.n
        print(json.dumps(run_one(m, ds, lang, n, args.threads, args.speed_only)), flush=True)
        return

    summary_path = OUT / ("speed.jsonl" if args.speed_only else "summary.jsonl")
    done = set()
    if summary_path.exists() and not args.force:
        for line in summary_path.read_text(encoding="utf-8").splitlines():
            s = json.loads(line)
            done.add((s["model"], s["dataset"], s["lang"]))

    todo = jobs(args.models.split(","), args.datasets.split(","), args.langs.split(","))
    for m, ds, lang in todo:
        if (m, ds, lang) in done:
            print(f"[skip] {m} {ds}-{lang}")
            continue
        print(f"[run ] {m} {ds}-{lang}", flush=True)
        cmd = [sys.executable, __file__, "--one", m, ds, lang, "--n", str(args.n),
               "--n-reazon", str(args.n_reazon), "--threads", str(args.threads),
               *(["--speed-only"] if args.speed_only else [])]
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                           env={**os.environ, "PYTHONUTF8": "1"})
        if p.returncode != 0:
            print(f"[fail] {m} {ds}-{lang}\n{p.stderr[-2000:]}", flush=True)
            continue
        result = json.loads(p.stdout.strip().splitlines()[-1])
        OUT.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")
        print(f"[done] {result}", flush=True)


if __name__ == "__main__":
    main()
