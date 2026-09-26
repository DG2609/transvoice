"""Collect results/asr and results/mt summaries into results/REPORT.md.

Accuracy comes from the full runs (summary.jsonl). Speed/CPU/RAM come from the idle-machine timing
pass (speed.jsonl) when it exists, because the accuracy runs were executed in parallel.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
TIMING_KEYS = ["rtf", "p50_ms", "p95_ms", "chunk_p50_ms", "chunk_p95_ms", "avg_cores", "model_ram_mb",
               "ram_mb", "tok_per_s"]


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _merge_speed(rows: list[dict], speed: list[dict], key) -> list[dict]:
    fast = {key(s): s for s in speed}
    out = []
    for r in rows:
        s = fast.get(key(r))
        out.append({**r, **{k: s[k] for k in TIMING_KEYS if s and k in s}, "timed_idle": bool(s)})
    return out


def _table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join("" if v is None else str(v) for v in r) + " |" for r in rows]
    return "\n".join(out)


def asr_section() -> str:
    rows = _merge_speed(_load(RES / "asr" / "summary.jsonl"), _load(RES / "asr" / "speed.jsonl"),
                        lambda r: (r["model"], r["dataset"], r["lang"]))
    parts = ["## ASR (CPU, 2 threads)\n",
             "Error = CER for Japanese, WER for English/Vietnamese (normalized, lower is better). "
             "RTF = processing time / audio time. Chunk p95 = compute per 560 ms chunk (streaming only). "
             "`*` = timing measured on an idle machine.\n"]
    for ds, lang in [("fleurs", "ja"), ("reazon", "ja"), ("fleurs", "en"), ("fleurs", "vi")]:
        sel = sorted((r for r in rows if r["dataset"] == ds and r["lang"] == lang), key=lambda r: r["error"])
        if not sel:
            continue
        parts.append(f"### {ds} / {lang} ({sel[0]['metric']}, n={sel[0]['n']})\n")
        parts.append(_table(
            ["model", sel[0]["metric"] + " %", "RTF", "p95 ms/utt", "chunk p95 ms", "RAM MB", "CPU cores"],
            [[r["model"] + ("*" if r["timed_idle"] else ""), r["error"], r["rtf"], r["p95_ms"],
              r.get("chunk_p95_ms"), r["model_ram_mb"], r["avg_cores"]] for r in sel]) + "\n")
    return "\n".join(parts)


def mt_section() -> str:
    key = lambda r: (r["engine"], r["src"], r["tgt"])  # noqa: E731
    speed = {key(r): r for r in _merge_speed(_load(RES / "mt" / "summary.jsonl"), _load(RES / "mt" / "speed.jsonl"), key)}
    comet = {key(r): r for r in _load(RES / "mt" / "comet_summary.jsonl")}
    parts = ["## Translation (CPU, 4 threads)\n",
             "COMET = wmt22-comet-da (higher is better). Bad % = COMET < 0.65 or a broken-output flag "
             "(empty, wrong language, too short/long, repetition, commentary). `*` = timing measured on an idle machine.\n"]
    for src, tgt in sorted({k[1:] for k in speed}):
        keys = sorted((k for k in speed if k[1:] == (src, tgt) and not k[0].startswith(("pipeline-", "edge-"))),
                      key=lambda k: -(comet.get(k, {}).get("comet") or 0))
        parts.append(f"### {src} → {tgt}\n")
        parts.append(_table(
            ["engine", "COMET", "bad %", "chrF", "flag %", "p50 ms", "p95 ms", "RAM MB", "CPU cores"],
            [[k[0] + ("*" if speed[k]["timed_idle"] else ""), comet.get(k, {}).get("comet"),
              comet.get(k, {}).get("bad_rate"), speed[k]["chrf"], speed[k]["flag_rate"], speed[k]["p50_ms"],
              speed[k]["p95_ms"], speed[k]["ram_mb"], speed[k]["avg_cores"]] for k in keys]) + "\n")
    return "\n".join(parts)


def lid_section() -> str:
    rows = _load(RES / "lid" / "summary.jsonl")
    if not rows:
        return ""
    return "## Spoken language ID (JA/EN/VI, silence trimmed)\n\n" + _table(
        ["detector", "audio heard", "accuracy %", "ja %", "en %", "vi %", "p50 ms"],
        [[r["detector"], r["audio"], r["accuracy"], r["ja"], r["en"], r["vi"], r["p50_ms"]] for r in rows]) + "\n"


def pipeline_section() -> str:
    comet = {(r["engine"], r["src"], r["tgt"]): r for r in _load(RES / "mt" / "comet_summary.jsonl")}
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((RES / "pipeline").glob("*.json"))]
    if not runs:
        return ""
    rows = []
    for r in runs:
        pairs = [comet.get((f"pipeline-{r['name']}", *p.split("-")), {}) for p in r["pairs"]]
        cs = [p["comet"] for p in pairs if p]
        bad = [p["bad_rate"] for p in pairs if p]
        rows.append([r["name"], r.get("cpus", "all"), " ".join(f"{k}:{v}" for k, v in r["asr_error"].items()),
                     round(sum(cs) / len(cs), 4) if cs else None, round(sum(bad) / len(bad), 1) if bad else None,
                     r["e2e_p50_ms"], r["e2e_p95_ms"], r["total_ram_mb"], r["avg_cores"]])
    return "## End-to-end: speech → ASR → translation (all 6 directions)\n\n" + _table(
        ["config", "CPUs", "ASR error %", "COMET avg", "bad % avg", "e2e p50 ms", "e2e p95 ms", "RAM MB", "CPU cores"],
        rows) + "\n"


def stability_section() -> str:
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((RES / "stability").glob("*.json"))]
    runs = [r for r in runs if r["name"] != "smoke"]
    if not runs:
        return ""
    rows = []
    for r in runs:
        tl = r["soak"]["ram_timeline"]
        # Allocators warm up during the first minutes; growth over the second half is what signals a leak.
        second_half = tl[-1][1] - tl[len(tl) // 2][1] if len(tl) > 2 else None
        rows.append([r["name"], r["soak"]["minutes"], r["soak"]["utterances"], r["soak"]["errors"],
                     r["soak"]["ram_start_mb"], r["soak"]["ram_max_mb"], second_half, r["soak"]["e2e_p99_ms"],
                     r["robustness"]["asr_hallucinated_clips"], r["robustness"]["mt_problem_cases"],
                     r["robustness"].get("mt_critical_errors")])
    return "## Stability\n\n" + _table(
        ["config", "minutes", "utterances", "errors", "RAM start MB", "RAM max MB", "RAM growth 2nd half MB",
         "e2e p99 ms", "ASR text on non-speech", "MT edge-case problems", "MT critical errors"], rows) + "\n"


NOTE = (
    "> All numbers are CPU-only (no GPU). Accuracy runs were executed in parallel, so their speed columns "
    "are pessimistic unless marked `*`. The **office-PC profile** (`CPUs = 12,13,14,15`: four E-cores of an "
    "i5-14600K, machine otherwise idle) is the reference for deployment speed.\n\n"
)


def main() -> None:
    text = ("# TransVoice CPU benchmark\n\n" + NOTE + asr_section() + "\n" + lid_section() + "\n" + mt_section()
            + "\n" + pipeline_section() + "\n" + stability_section())
    (RES / "REPORT.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
