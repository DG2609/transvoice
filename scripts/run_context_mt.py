"""Does the previous sentence help the translation? Business Scene Dialogue (BSD, JA<->EN dialogues with
reference translations), each sentence translated alone and with the previous 1-2 source sentences as context
(HY-MT's own contextual prompt).

  python scripts/run_context_mt.py --n 300 --cpus 0,1,2,3,4,5
  .venv-comet/Scripts/python scripts/run_comet.py    # scores results/mt/bsd-*.jsonl

Writes results/mt/bsd-<variant>__<src>-<tgt>.jsonl.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from bench.data import ROOT  # noqa: E402
from bench.metrics import corpus_chrf, translation_flags  # noqa: E402
from bench.resources import limit_cpus  # noqa: E402
from transvoice.engine import DEFAULT_MT, FAST_MT_ARGS  # noqa: E402
from transvoice.mt import LlamaEngine, make_hy_mt_prompt  # noqa: E402

COLUMN = {"ja": "ja_sentence", "en": "en_sentence"}


def dialogues(src: str, n: int) -> list[list[dict]]:
    """Whole dialogues written originally in `src`, in order, until about n sentences."""
    df = pd.read_parquet(ROOT / "data" / "bsd" / "test.parquet")
    out, total = [], 0
    for _, d in df[df.original_language == src].groupby("id", sort=True):
        out.append(d.sort_values("no").to_dict("records"))
        total += len(d)
        if total >= n:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="sentences per direction (whole dialogues)")
    ap.add_argument("--pairs", default="ja-en,en-ja")
    ap.add_argument("--contexts", default="0,1,2", help="previous sentences given as context")
    ap.add_argument("--mode", default="history", choices=("history", "prompt"),
                    help="history: earlier sentences and their translations as chat turns; "
                         "prompt: HY-MT's contextual prompt")
    ap.add_argument("--cpus")
    args = ap.parse_args()
    limit_cpus(args.cpus)

    engine = LlamaEngine(DEFAULT_MT, make_hy_mt_prompt(), 4, reuse_prefix=True, extra_args=FAST_MT_ARGS)
    engine.translate("Hello.", "en", "ja")
    for pair in args.pairs.split(","):
        src, tgt = pair.split("-")
        dias = dialogues(src, args.n)
        for k in map(int, args.contexts.split(",")):
            rows, t0 = [], time.perf_counter()
            for d in dias:
                done = []  # (source, our own translation) of this dialogue so far
                for i, r in enumerate(d):
                    if args.mode == "prompt":
                        context = [x[COLUMN[src]] for x in d[max(0, i - k):i]] if k else []
                        tr = engine.translate(r[COLUMN[src]], src, tgt, context=context)
                    else:
                        context = done[-k:] if k else []
                        tr = engine.translate(r[COLUMN[src]], src, tgt, history=context)
                    done.append((r[COLUMN[src]], tr.text))
                    rows.append({"id": f"{r['id']}#{r['no']}", "src": r[COLUMN[src]], "ref": r[COLUMN[tgt]],
                                 "hyp": tr.text, "context": context,
                                 "flags": translation_flags(r[COLUMN[src]], tr.text, r[COLUMN[tgt]], tgt)})
            secs = time.perf_counter() - t0
            name = f"bsd-ctx{k}" if args.mode == "prompt" or not k else f"bsd-hist{k}"
            with open(ROOT / "results" / "mt" / f"{name}__{src}-{tgt}.jsonl", "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            chrf = corpus_chrf([r["ref"] for r in rows], [r["hyp"] for r in rows])
            print(f"{name} {src}->{tgt}: {len(rows)} sentences, chrF {chrf:.2f}, "
                  f"{1000 * secs / len(rows):.0f} ms/sentence", flush=True)
    engine.close()


if __name__ == "__main__":
    main()
