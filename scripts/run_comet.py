"""Score every results/mt/*.jsonl with COMET (Unbabel/wmt22-comet-da), CPU.

Runs in the separate COMET environment (it needs numpy<2):
  .venv-comet/Scripts/python.exe scripts/run_comet.py
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "hf"))

from comet import load_from_checkpoint  # noqa: E402

from bench.metrics import translation_flags  # noqa: E402

MT = ROOT / "results" / "mt"
CKPT = ROOT / "models" / "eval" / "wmt22-comet-da" / "checkpoints" / "model.ckpt"
# A translation counts as "bad" if a heuristic flag fired or COMET falls below this.
BAD_THRESHOLD = 0.65


def main() -> None:
    model = load_from_checkpoint(str(CKPT))
    summary = []
    for f in sorted(MT.glob("*__*.jsonl")):
        rows = [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines()]
        if not rows:
            continue
        engine, pair = f.stem.split("__")
        src, tgt = pair.split("-")
        if not all("comet" in r for r in rows):
            # For speech pipelines "src" is the ASR output; score against the true source text.
            data = [{"src": r.get("src_gold", r["src"]), "mt": r["hyp"], "ref": r["ref"]} for r in rows]
            pred = model.predict(data, batch_size=16, gpus=0, progress_bar=False)
            for r, s in zip(rows, pred.scores):
                r["comet"] = round(float(s), 4)
        if not engine.startswith("edge-"):  # edge-case files carry their own critical-error flags
            for r in rows:
                r["flags"] = translation_flags(r.get("src_gold", r["src"]), r["hyp"], r["ref"], tgt)
        f.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        bad = [bool(r["flags"]) or r["comet"] < BAD_THRESHOLD for r in rows]
        summary.append({
            "engine": engine, "src": src, "tgt": tgt, "n": len(rows),
            "comet": round(sum(r["comet"] for r in rows) / len(rows), 4),
            "bad_rate": round(100 * sum(bad) / len(rows), 1),
        })
        print(json.dumps(summary[-1]), flush=True)
    (MT / "comet_summary.jsonl").write_text("".join(json.dumps(s) + "\n" for s in summary), encoding="utf-8")


if __name__ == "__main__":
    main()
