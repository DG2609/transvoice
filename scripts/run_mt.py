"""Benchmark translation engines on FLEURS parallel sentences (CPU only).

  python scripts/run_mt.py                          # all engines, all supported directions
  python scripts/run_mt.py --engines nllb-600m --n 20
  python scripts/run_mt.py --source asr:sensevoice  # translate ASR output instead of gold text
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import psutil  # noqa: E402

from bench.data import ROOT, fleurs_parallel  # noqa: E402
from bench.metrics import corpus_chrf, translation_flags  # noqa: E402
from transvoice.mt import REGISTRY  # noqa: E402
from bench.resources import ResourceMonitor  # noqa: E402

OUT = ROOT / "results" / "mt"


def asr_sources(asr_model: str, lang: str) -> dict[int, str]:
    """FLEURS sentence id -> ASR hypothesis, from a previous run_asr.py result file."""
    path = ROOT / "results" / "asr" / f"{asr_model}__fleurs-{lang}.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return {int(r["key"].rsplit("-", 1)[1]): r["hyp"] for r in rows}


def run_one(engine_name: str, n: int, threads: int, source: str, speed_only: bool = False) -> list[dict]:
    spec = REGISTRY[engine_name]
    sents = fleurs_parallel(n)
    tag = "" if source == "gold" else "@" + source.replace(":", "-")

    me = psutil.Process(os.getpid())
    base_rss = me.memory_info().rss / 2**20
    engine = spec.build(threads)
    engine.translate("Hello.", "en", "ja")  # warm-up
    summaries = []
    try:
        for src, tgt in spec.pairs:
            items = sents
            if source != "gold":
                hyps = asr_sources(source.split(":", 1)[1], src)
                items = [dict(s, **{src: hyps[s["id"]]}) for s in sents if s["id"] in hyps]
            rows = []
            in_process = not engine.pids
            with ResourceMonitor([os.getpid()] if in_process else engine.pids) as mon:
                for s in items:
                    tr = engine.translate(s[src], src, tgt)
                    rows.append({
                        "id": s["id"], "src": s[src], "ref": s[tgt], "hyp": tr.text,
                        "latency_s": round(tr.latency_s, 3), "out_tokens": tr.out_tokens,
                        "flags": translation_flags(s[src], tr.text, s[tgt], tgt),
                    })
            OUT.mkdir(parents=True, exist_ok=True)
            if not speed_only:
                with open(OUT / f"{engine_name}{tag}__{src}-{tgt}.jsonl", "w", encoding="utf-8") as f:
                    for r in rows:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
            lat = [r["latency_s"] for r in rows]
            tok = [r["out_tokens"] / r["latency_s"] for r in rows if r["out_tokens"] and r["latency_s"] > 0]
            summaries.append({
                "engine": engine_name + tag, "src": src, "tgt": tgt, "n": len(rows),
                "chrf": round(corpus_chrf([r["ref"] for r in rows], [r["hyp"] for r in rows]), 2),
                "flag_rate": round(100 * sum(bool(r["flags"]) for r in rows) / len(rows), 1),
                "p50_ms": round(1000 * statistics.median(lat)),
                "p95_ms": round(1000 * float(np.percentile(lat, 95))),
                "tok_per_s": round(statistics.median(tok), 1) if tok else None,
                "avg_cores": round(mon.avg_cores, 2),
                "ram_mb": round(mon.peak_rss_mb - base_rss if in_process else mon.peak_rss_mb),
                "threads": threads,
            })
    finally:
        engine.close()
    return summaries


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", default=",".join(REGISTRY))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--source", default="gold", help="'gold' or 'asr:<model>'")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--speed-only", action="store_true",
                    help="timing pass on an idle machine: no per-sentence files, results go to speed.jsonl")
    ap.add_argument("--one", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.one:
        for s in run_one(args.one, args.n, args.threads, args.source, args.speed_only):
            print("RESULT " + json.dumps(s), flush=True)
        return

    summary_path = OUT / ("speed.jsonl" if args.speed_only else "summary.jsonl")
    done = set()
    if summary_path.exists() and not args.force:
        done = {json.loads(line)["engine"] for line in summary_path.read_text(encoding="utf-8").splitlines()}
    tag = "" if args.source == "gold" else "@" + args.source.replace(":", "-")
    for name in args.engines.split(","):
        if name + tag in done:
            print(f"[skip] {name}{tag}")
            continue
        print(f"[run ] {name}{tag}", flush=True)
        cmd = [sys.executable, __file__, "--one", name, "--n", str(args.n), "--threads", str(args.threads),
               "--source", args.source, *(["--speed-only"] if args.speed_only else [])]
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                           env={**os.environ, "PYTHONUTF8": "1"})
        results = [json.loads(line[7:]) for line in p.stdout.splitlines() if line.startswith("RESULT ")]
        if p.returncode != 0 or not results:
            print(f"[fail] {name}\n{p.stderr[-2000:]}", flush=True)
            continue
        OUT.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "a", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        for r in results:
            print(f"[done] {r}", flush=True)


if __name__ == "__main__":
    main()
