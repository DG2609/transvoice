"""End-to-end quality of the live app: FLEURS speech is played through the real engine at real-time speed
(chunking, drafts, sentence ends, re-recognition), and the FINAL subtitles are scored.

  python scripts/run_live_eval.py --name base --cpus 12,13,14,15
  python scripts/run_live_eval.py --name no-rescore --no-rescore
  .venv-comet/Scripts/python scripts/run_comet.py      # adds COMET for results/mt/live-<name>__*.jsonl

Writes results/live/<name>.json (ASR error of the final text, latency, drafts per sentence) and
results/mt/live-<name>__<src>-<tgt>.jsonl (one row per utterance, for COMET).
"""
import argparse
import json
import statistics
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import psutil  # noqa: E402

from bench.data import ROOT, fleurs_parallel, fleurs_utterances  # noqa: E402
from bench.metrics import asr_error_rate, translation_flags  # noqa: E402
from bench.resources import limit_cpus  # noqa: E402
from transvoice.engine import Engine, Settings  # noqa: E402
from transvoice.paths import SAMPLE_RATE  # noqa: E402

GAP_S = 2.5  # silence between utterances, so every utterance ends its sentences


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--me", default="vi")
    ap.add_argument("--sources", default="ja,en")
    ap.add_argument("--n", type=int, default=15, help="utterances per source language")
    ap.add_argument("--cpus")
    ap.add_argument("--no-rescore", action="store_true")
    ap.add_argument("--mt-fast", action=argparse.BooleanOptionalAction, default=None,
                    help="override Settings.mt_fast (default: the app default)")
    ap.add_argument("--no-drafts", action="store_true")
    ap.add_argument("--mt-threads", type=int)
    ap.add_argument("--asr-threads", type=int)
    ap.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE",
                    help="override a numeric Settings field, e.g. --set sentence_gap_s=0.5")
    args = ap.parse_args()
    limit_cpus(args.cpus)

    finals, drafts, lock = [], {}, threading.Lock()

    def on_event(kind, s):
        if kind == "translated":
            with lock:
                if s.final:
                    finals.append((time.time(), s))
                else:
                    drafts[s.id] = drafts.get(s.id, 0) + 1

    overrides = {} if args.mt_fast is None else {"mt_fast": args.mt_fast}
    if args.mt_threads:
        overrides["mt_threads"] = args.mt_threads
    if args.asr_threads:
        overrides["asr_threads"] = args.asr_threads
    for item in args.set:
        key, value = item.split("=", 1)
        overrides[key] = float(value)
    settings = Settings(my_lang=args.me, their_lang="auto", rescore_final=not args.no_rescore,
                        drafts=not args.no_drafts, **overrides)
    engine = Engine(settings, on_event)
    engine.start()

    # Load from other programs skews the timings on a shared machine: sample the whole machine's CPU use.
    load, stop_load = [], threading.Event()

    def sample_load() -> None:
        while not stop_load.is_set():
            load.append(psutil.cpu_percent(interval=1.0))

    threading.Thread(target=sample_load, daemon=True).start()

    sents = fleurs_parallel(args.n, seed=7)
    utts = {lang: fleurs_utterances(lang) for lang in args.sources.split(",")}
    windows = []  # (lang, sentence dict, start, end) in wall-clock time
    chunk = SAMPLE_RATE // 10
    for lang in utts:
        for s in sents:
            audio = np.concatenate([utts[lang][s["id"]].load(), np.zeros(int(GAP_S * SAMPLE_RATE), np.float32)])
            start = time.time()
            for i in range(0, len(audio), chunk):
                engine.feed("them", audio[i : i + chunk])
                time.sleep(max(0.0, start + (i + chunk) / SAMPLE_RATE - time.time()))
            windows.append((lang, s, start, time.time() - GAP_S))
    engine.flush()
    engine.wait_idle()
    engine.stop()
    stop_load.set()

    out_rows, per_lang, detail = {}, {}, []
    lags, stages, extra_mt = [], {"close_ms": [], "final_start_ms": [], "final_first_ms": [], "final_mt_ms": []}, 0
    for lang, s, start, end in windows:
        # In speaking order (as the overlay shows them), not in the order the finals arrived.
        mine = sorted(((t, f) for t, f in finals if start - 0.5 <= f.started_at <= end + 0.5), key=lambda x: x[1].id)
        text = ("" if lang == "ja" else " ").join(f.text for _, f in mine)
        translation = " ".join(f.translation or "" for _, f in mine).strip()
        lags += [f.timings.get("lag_ms", 0) for _, f in mine if f.translation]
        for _, f in mine:
            for k in stages:
                if k in f.timings:
                    stages[k].append(f.timings[k])
            extra_mt += "final_mt_ms" in f.timings  # the final needed a translation after the sentence closed
            detail.append({"lang": lang, "id": f.id, "audio_s": round(f.audio_s, 2), "chunks": len(f.chunks),
                           "drafts": drafts.get(f.id, 0), "text": f.text, **f.timings})
        per_lang.setdefault(lang, {"refs": [], "hyps": [], "sentences": 0, "drafts": 0})
        pl = per_lang[lang]
        pl["refs"].append(s[lang])
        pl["hyps"].append(text)
        pl["sentences"] += len(mine)
        pl["drafts"] += sum(drafts.get(f.id, 0) for _, f in mine)
        if lang != args.me:
            out_rows.setdefault((lang, args.me), []).append({
                "id": s["id"], "src": text, "src_gold": s[lang], "ref": s[args.me], "hyp": translation,
                "flags": translation_flags(s[lang], translation, s[args.me], args.me)})

    mt_dir = ROOT / "results" / "mt"
    for (src, tgt), rows in out_rows.items():
        with open(mt_dir / f"live-{args.name}__{src}-{tgt}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {
        "name": args.name, "cpus": args.cpus or "all", "rescore": settings.rescore_final,
        "mt_fast": settings.mt_fast, "drafts": settings.drafts, "mt_model": Path(settings.mt_model).name,
        "mt_threads": settings.mt_threads, "asr_threads": settings.asr_threads, "set": args.set,
        "asr_error": {l: round(asr_error_rate(v["refs"], v["hyps"], l), 2) for l, v in per_lang.items()},
        "sentences_per_utt": {l: round(v["sentences"] / len(v["refs"]), 2) for l, v in per_lang.items()},
        "drafts_per_sentence": {l: round(v["drafts"] / max(1, v["sentences"]), 2) for l, v in per_lang.items()},
        "final_lag_p50_ms": round(statistics.median(lags)) if lags else None,
        "final_lag_p90_ms": round(float(np.percentile(lags, 90))) if lags else None,
        # where the final lag goes (medians): pause detection, waiting for the translator, translating
        "stages_ms": {k: round(statistics.median(v)) for k, v in stages.items() if v},
        "final_needed_extra_mt": f"{extra_mt}/{len(lags)}",
        "close_reasons": {r: sum(d.get("close") == r for d in detail) for r in sorted({d.get("close") for d in detail} - {None})},
        "machine_cpu_avg_pct": round(statistics.mean(load)) if load else None,
    }
    (ROOT / "results" / "live").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "live" / f"{args.name}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with open(ROOT / "results" / "live" / f"{args.name}.sentences.jsonl", "w", encoding="utf-8") as f:
        for d in detail:  # per-sentence timings, for digging into the lag
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
