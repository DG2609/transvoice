"""End-to-end test of a full app configuration: speech -> ASR -> translation, all models loaded at once.

  python scripts/run_pipeline.py --name lite \
      --asr ja=sensevoice,en=sensevoice,vi=zipformer-vi-30m --mt default=nllb-1.3b

  --mt accepts per-pair routing, e.g. "ja-en=cat-translate-1.4b,en-ja=cat-translate-1.4b,default=hy-mt-1.8b".
Writes results/mt/pipeline-<name>__<src>-<tgt>.jsonl (scored later by run_comet.py) and
results/pipeline/<name>.json with end-to-end latency, CPU and total RAM.
"""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import psutil  # noqa: E402

from transvoice.asr import REGISTRY as ASR  # noqa: E402
from bench.data import ROOT, fleurs_parallel, fleurs_utterances  # noqa: E402
from bench.metrics import asr_error_rate, corpus_chrf, translation_flags  # noqa: E402
from transvoice.mt import REGISTRY as MT  # noqa: E402
from bench.resources import ResourceMonitor, limit_cpus  # noqa: E402
from transvoice.textnorm import asr_text_for_mt  # noqa: E402

LANGS = ("ja", "en", "vi")


def parse_map(spec: str) -> dict[str, str]:
    return dict(item.split("=", 1) for item in spec.split(","))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--asr", required=True, help="lang=model,...")
    ap.add_argument("--mt", required=True, help="src-tgt=engine,...,default=engine")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--asr-threads", type=int, default=2)
    ap.add_argument("--mt-threads", type=int, default=4)
    ap.add_argument("--cpus", help="pin to these logical CPUs, e.g. 12,13,14,15 (E-cores = office-PC profile)")
    args = ap.parse_args()
    limit_cpus(args.cpus)

    asr_map, mt_map = parse_map(args.asr), parse_map(args.mt)
    sents = fleurs_parallel(args.n)
    utts = {lang: fleurs_utterances(lang) for lang in LANGS}
    audio = {(lang, s["id"]): utts[lang][s["id"]].load() for lang in LANGS for s in sents}

    me = psutil.Process(os.getpid())
    base_rss = me.memory_info().rss / 2**20
    asr_models = {}
    for lang in LANGS:
        name = asr_map[lang]
        asr_models[lang] = ASR[name].build(lang, args.asr_threads)
    engines = {name: MT[name].build(args.mt_threads) for name in set(mt_map.values())}
    for e in engines.values():
        e.translate("Hello.", "en", "ja")

    def engine_for(src, tgt):
        return mt_map.get(f"{src}-{tgt}", mt_map["default"])

    pids = [os.getpid()] + [p for e in engines.values() for p in e.pids]
    out_dir = ROOT / "results" / "mt"
    rows = {(s, t): [] for s in LANGS for t in LANGS if s != t}
    asr_refs, asr_hyps, asr_ms, mt_ms, e2e_ms = {l: [] for l in LANGS}, {l: [] for l in LANGS}, [], [], []
    try:
        with ResourceMonitor(pids) as mon:
            for s in sents:
                for src in LANGS:
                    t0 = time.perf_counter()
                    hyp = asr_models[src].transcribe(audio[(src, s["id"])])
                    t_asr = time.perf_counter() - t0
                    asr_ms.append(1000 * t_asr)
                    asr_refs[src].append(utts[src][s["id"]].text)
                    asr_hyps[src].append(hyp)
                    for tgt in LANGS:
                        if tgt == src:
                            continue
                        tr = engines[engine_for(src, tgt)].translate(asr_text_for_mt(hyp), src, tgt)
                        mt_ms.append(1000 * tr.latency_s)
                        e2e_ms.append(1000 * (t_asr + tr.latency_s))
                        rows[(src, tgt)].append({
                            "id": s["id"], "src": hyp, "src_gold": s[src], "ref": s[tgt], "hyp": tr.text,
                            "asr_ms": round(1000 * t_asr), "mt_ms": round(1000 * tr.latency_s),
                            "flags": translation_flags(s[src], tr.text, s[tgt], tgt),
                        })
    finally:
        for e in engines.values():
            e.close()

    out_dir.mkdir(parents=True, exist_ok=True)
    per_pair = {}
    for (src, tgt), rs in rows.items():
        with open(out_dir / f"pipeline-{args.name}__{src}-{tgt}.jsonl", "w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        per_pair[f"{src}-{tgt}"] = {
            "engine": engine_for(src, tgt),
            "chrf": round(corpus_chrf([r["ref"] for r in rs], [r["hyp"] for r in rs]), 2),
            "flag_rate": round(100 * sum(bool(r["flags"]) for r in rs) / len(rs), 1),
        }
    summary = {
        "name": args.name, "asr": asr_map, "mt": mt_map, "cpus": args.cpus or "all", "n_sentences": len(sents),
        "asr_error": {l: round(asr_error_rate(asr_refs[l], asr_hyps[l], l), 2) for l in LANGS},
        "asr_p50_ms": round(statistics.median(asr_ms)), "asr_p95_ms": round(float(np.percentile(asr_ms, 95))),
        "mt_p50_ms": round(statistics.median(mt_ms)), "mt_p95_ms": round(float(np.percentile(mt_ms, 95))),
        "e2e_p50_ms": round(statistics.median(e2e_ms)), "e2e_p95_ms": round(float(np.percentile(e2e_ms, 95))),
        "avg_cores": round(mon.avg_cores, 2),
        "total_ram_mb": round(mon.peak_rss_mb - base_rss),
        "pairs": per_pair,
    }
    (ROOT / "results" / "pipeline").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "pipeline" / f"{args.name}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
