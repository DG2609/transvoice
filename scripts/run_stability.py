"""Stability test for one full app configuration (CPU only).

1. Robustness: ASR on non-speech audio must stay (near) empty; MT on awkward inputs must not
   invent text, switch language or loop.
2. Soak: speech -> ASR -> translation into the other two languages, non-stop for --minutes,
   sampling RAM every few seconds. Reports RAM drift, latency tail and error count.

  python scripts/run_stability.py --name lite --asr ja=sensevoice,en=sensevoice,vi=zipformer-vi-30m \
      --mt default=qwen3.5-2b --minutes 20
"""
import argparse
import itertools
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import psutil  # noqa: E402

from transvoice.asr import REGISTRY as ASR  # noqa: E402
from bench.data import ROOT, SAMPLE_RATE, fleurs_utterances  # noqa: E402
from bench.metrics import _has_repetition  # noqa: E402
from transvoice.mt import REGISTRY as MT  # noqa: E402
from bench.resources import limit_cpus  # noqa: E402
from transvoice.textnorm import asr_text_for_mt, normalize, script_ok  # noqa: E402

LANGS = ("ja", "en", "vi")
OUT = ROOT / "results" / "stability"

# Inputs an interpreter app meets in real calls: backchannels, fragments, numbers, code-mixing.
# "ref" = reference translations (scored with COMET later); "must" = facts the output has to keep
# (every group needs one match), e.g. the weekday or a number. Missing one is a critical error.
MT_EDGE_CASES = [
    {"src": "ja", "text": "はい。", "ref": {"en": "Yes.", "vi": "Vâng."}},
    {"src": "ja", "text": "えーと、", "ref": {"en": "Um,", "vi": "Ờ,"}},
    {"src": "ja", "text": "そうですね", "ref": {"en": "That's right.", "vi": "Đúng vậy."}},
    {"src": "ja", "text": "2026年9月25日午後3時", "ref": {"en": "September 25, 2026, 3 p.m.", "vi": "3 giờ chiều ngày 25 tháng 9 năm 2026"},
     "must": {"en": [["2026"], ["25"], ["3 p", "3:00", "3pm", "3 pm", "15:00"], ["September", "Sept", "9/"]],
              "vi": [["2026"], ["25"], ["3 giờ", "15 giờ", "15h", "3h", "15:00"], ["tháng 9", "/9", "-9"]]}},
    {"src": "ja", "text": "このmeetingはrescheduleしましょう",
     "ref": {"en": "Let's reschedule this meeting.", "vi": "Chúng ta dời lịch cuộc họp này nhé."}},
    {"src": "ja", "text": "あの、その、例の件なんですけど",
     "ref": {"en": "Um, so, about that matter we talked about...", "vi": "À, ừm, về chuyện hôm trước ấy..."}},
    {"src": "ja", "text": "来週の月曜日は休みです。", "ref": {"en": "Next Monday is a day off.", "vi": "Thứ hai tuần sau được nghỉ."},
     "must": {"en": [["Monday"]], "vi": [["thứ hai", "thứ 2"]]}},
    {"src": "en", "text": "OK.", "ref": {"ja": "わかりました。", "vi": "Được."}},
    {"src": "en", "text": "Uh-huh.", "ref": {"ja": "うん。", "vi": "Ừ."}},
    {"src": "en", "text": "So, um, basically", "ref": {"ja": "それで、えーと、基本的には", "vi": "Thì, ừm, về cơ bản là"}},
    {"src": "en", "text": "The Q3 KPI is 12.5% YoY.", "ref": {"ja": "第3四半期のKPIは前年比12.5%です。", "vi": "KPI quý 3 tăng 12,5% so với cùng kỳ năm trước."},
     "must": {"ja": [["12.5", "12,5", "１２．５"]], "vi": [["12,5", "12.5"]]}},
    {"src": "en", "text": "We can't ship before Friday.", "ref": {"ja": "金曜日より前には出荷できません。", "vi": "Chúng tôi không thể giao hàng trước thứ Sáu."},
     "must": {"ja": [["金曜"], ["ない", "ません", "できず"]], "vi": [["thứ sáu", "thứ 6"], ["không"]]}},
    {"src": "vi", "text": "Dạ.", "ref": {"ja": "はい。", "en": "Yes."}},
    {"src": "vi", "text": "Vâng ạ", "ref": {"ja": "はい、そうです。", "en": "Yes."}},
    {"src": "vi", "text": "ờ thì", "ref": {"ja": "えーと", "en": "Well, um"}},
    {"src": "vi", "text": "cuộc họp dời sang thứ năm nhé anh", "ref": {"ja": "会議は木曜日に変更になりますね。", "en": "The meeting's been moved to Thursday, okay?"},
     "must": {"ja": [["木曜"]], "en": [["Thursday"]]}},
    {"src": "vi", "text": "Giá là 1,5 triệu đồng, không phải 15 triệu.", "ref": {"ja": "価格は150万ドンで、1500万ドンではありません。", "en": "The price is 1.5 million dong, not 15 million."},
     "must": {"ja": [["150万", "1.5百万", "1,5"]], "en": [["1.5 million", "1,500,000", "1.5m"]]}},
]


def missing_facts(output: str, groups: list[list[str]]) -> list[str]:
    low = output.lower()
    return [" / ".join(g) for g in groups if not any(alt.lower() in low for alt in g)]


def parse_map(spec: str) -> dict[str, str]:
    return dict(item.split("=", 1) for item in spec.split(","))


def non_speech_clips() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    t = np.arange(5 * SAMPLE_RATE) / SAMPLE_RATE
    return {
        "silence_10s": np.zeros(10 * SAMPLE_RATE, np.float32),
        "quiet_noise_10s": (0.01 * rng.standard_normal(10 * SAMPLE_RATE)).astype(np.float32),
        "loud_noise_10s": (0.1 * rng.standard_normal(10 * SAMPLE_RATE)).astype(np.float32),
        "beep_5s": (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32),
        "hum_5s": (0.2 * np.sin(2 * np.pi * 50 * t) + 0.05 * rng.standard_normal(len(t))).astype(np.float32),
    }


def robustness(asr_models: dict, engines: dict, engine_for, name: str) -> dict:
    asr_rows = []
    for clip_name, clip in non_speech_clips().items():
        for lang, asr in asr_models.items():
            text = asr.transcribe(clip)
            asr_rows.append({"clip": clip_name, "lang": lang, "output": text, "chars": len(normalize(text, lang))})
    mt_rows = []
    for case in MT_EDGE_CASES:
        src, text = case["src"], case["text"]
        for tgt, ref in case["ref"].items():
            out = engines[engine_for(src, tgt)].translate(text, src, tgt).text
            ratio = len(normalize(out, tgt)) / max(1, len(normalize(text, src)))
            problems = []
            if not out.strip():
                problems.append("empty")
            elif not script_ok(out, tgt) and len(normalize(out, tgt)) > 3:
                problems.append("wrong_language")
            if ratio > 4 and len(out) > 20:
                problems.append("invented_text")
            if _has_repetition(out, tgt):
                problems.append("repetition")
            if "\n" in out:
                problems.append("multiline")
            missing = missing_facts(out, case.get("must", {}).get(tgt, []))
            if missing:
                problems.append("critical:" + ",".join(missing))
            mt_rows.append({"src": src, "tgt": tgt, "input": text, "ref": ref, "output": out, "problems": problems})
    # Same row format as run_mt.py so run_comet.py scores the edge cases too.
    mt_dir = ROOT / "results" / "mt"
    for src in LANGS:
        for tgt in LANGS:
            rows = [r for r in mt_rows if (r["src"], r["tgt"]) == (src, tgt)]
            if rows:
                with open(mt_dir / f"edge-{name}__{src}-{tgt}.jsonl", "w", encoding="utf-8") as f:
                    for r in rows:
                        f.write(json.dumps({"src": r["input"], "ref": r["ref"], "hyp": r["output"],
                                            "flags": r["problems"]}, ensure_ascii=False) + "\n")
    return {
        "asr_non_speech": asr_rows,
        "asr_hallucinated_clips": sum(r["chars"] > 3 for r in asr_rows),
        "mt_edge_cases": mt_rows,
        "mt_problem_cases": sum(bool(r["problems"]) for r in mt_rows),
        "mt_critical_errors": sum(any(p.startswith("critical") for p in r["problems"]) for r in mt_rows),
    }


def rss_mb(procs: list[psutil.Process]) -> float:
    total = 0
    for p in procs:
        try:
            total += p.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    return total / 2**20


def soak(asr_models: dict, engines: dict, engine_for, minutes: float, pids: list[int]) -> dict:
    utts = {lang: list(fleurs_utterances(lang).values())[:60] for lang in LANGS}
    audio = {lang: [u.load() for u in us] for lang, us in utts.items()}
    procs = [psutil.Process(p) for p in pids]
    samples, stop = [], threading.Event()
    t_start = time.perf_counter()

    def sampler():
        while not stop.is_set():
            samples.append((time.perf_counter() - t_start, rss_mb(procs)))
            stop.wait(5)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    e2e, errors, iterations, speech_s = [], [], 0, 0.0
    order = itertools.cycle([(lang, i) for i in range(60) for lang in LANGS])
    deadline = t_start + 60 * minutes
    while time.perf_counter() < deadline:
        lang, i = next(order)
        try:
            t0 = time.perf_counter()
            text = asr_models[lang].transcribe(audio[lang][i])
            t_asr = time.perf_counter() - t0
            for tgt in LANGS:
                if tgt != lang:
                    tr = engines[engine_for(lang, tgt)].translate(asr_text_for_mt(text) or ".", lang, tgt)
                    e2e.append(1000 * (t_asr + tr.latency_s))
            speech_s += utts[lang][i].duration
        except Exception:  # noqa: BLE001 - the point is to count failures
            errors.append(traceback.format_exc(limit=3))
        iterations += 1
    stop.set()
    th.join()

    ts = np.array([s[0] for s in samples])
    mem = np.array([s[1] for s in samples])
    # RAM drift: compare the last 20% of the run with the first 20% after warm-up.
    k = max(1, len(mem) // 5)
    drift = float(mem[-k:].mean() - mem[k : 2 * k].mean()) if len(mem) >= 3 * k else float("nan")
    slope = float(np.polyfit(ts[k:], mem[k:], 1)[0] * 3600) if len(mem) > k + 2 else float("nan")
    return {
        "minutes": minutes, "utterances": iterations, "speech_minutes_processed": round(speech_s / 60, 1),
        "errors": len(errors), "error_samples": errors[:3],
        "ram_start_mb": round(float(mem[0])), "ram_max_mb": round(float(mem.max())), "ram_end_mb": round(float(mem[-1])),
        "ram_drift_mb": round(drift, 1), "ram_slope_mb_per_hour": round(slope, 1),
        "e2e_p50_ms": round(float(np.percentile(e2e, 50))), "e2e_p95_ms": round(float(np.percentile(e2e, 95))),
        "e2e_p99_ms": round(float(np.percentile(e2e, 99))), "e2e_max_ms": round(float(max(e2e))),
        "ram_timeline": [(round(t), round(m)) for t, m in samples],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--asr", required=True)
    ap.add_argument("--mt", required=True)
    ap.add_argument("--minutes", type=float, default=20)
    ap.add_argument("--asr-threads", type=int, default=2)
    ap.add_argument("--mt-threads", type=int, default=4)
    ap.add_argument("--cpus", help="pin to these logical CPUs, e.g. 12,13,14,15 (E-cores = office-PC profile)")
    args = ap.parse_args()
    limit_cpus(args.cpus)

    asr_map, mt_map = parse_map(args.asr), parse_map(args.mt)
    asr_models = {lang: ASR[asr_map[lang]].build(lang, args.asr_threads) for lang in LANGS}
    engines = {name: MT[name].build(args.mt_threads) for name in set(mt_map.values())}

    def engine_for(src, tgt):
        return mt_map.get(f"{src}-{tgt}", mt_map["default"])

    pids = [os.getpid()] + [p for e in engines.values() for p in e.pids]
    try:
        result = {"name": args.name, "asr": asr_map, "mt": mt_map,
                  "robustness": robustness(asr_models, engines, engine_for, args.name),
                  "soak": soak(asr_models, engines, engine_for, args.minutes, pids)}
    finally:
        for e in engines.values():
            e.close()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{args.name}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    s, r = result["soak"], result["robustness"]
    print(json.dumps({"name": args.name, "asr_hallucinated_clips": r["asr_hallucinated_clips"],
                      "mt_problem_cases": r["mt_problem_cases"], "mt_critical_errors": r["mt_critical_errors"],
                      **{k: v for k, v in s.items() if k not in ("ram_timeline", "error_samples")}}), flush=True)


if __name__ == "__main__":
    main()
