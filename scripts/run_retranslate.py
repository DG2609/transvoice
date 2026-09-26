"""Speed of live re-translation: every sentence is translated again each time a chunk is appended.

Replays the chunk sequences of a saved session and compares llama-server settings. Greedy speculative
decoding must not change the output, so every variant is checked against the baseline text.

  python scripts/run_retranslate.py sessions/<session>.jsonl --cpus 12,13,14,15
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.resources import limit_cpus  # noqa: E402
from transvoice.mt import LlamaEngine, MT_DIR, make_hy_mt_prompt  # noqa: E402

VARIANTS = {
    "baseline": dict(reuse_prefix=False, extra_args=()),
    "prefix-reuse": dict(reuse_prefix=True, extra_args=()),
    "prefix+ngram-cache": dict(reuse_prefix=True, extra_args=("--spec-type", "ngram-cache")),
    "prefix+ngram-simple": dict(reuse_prefix=True, extra_args=("--spec-type", "ngram-simple")),
    "prefix+ngram-mod": dict(reuse_prefix=True, extra_args=("--spec-type", "ngram-mod")),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("session", type=Path)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--cpus")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    args = ap.parse_args()
    limit_cpus(args.cpus)

    rows = [json.loads(line) for line in args.session.read_text(encoding="utf-8").splitlines()]
    sentences = [r for r in rows if r["translation"] and len(r["chunks"]) >= 2]
    joiner = lambda r, k: ("" if r["lang"] == "ja" else " ").join(r["chunks"][:k])  # noqa: E731
    jobs = [(joiner(r, k), r["lang"], r["target"]) for r in sentences for k in range(1, len(r["chunks"]) + 1)]
    print(f"{len(sentences)} sentences, {len(jobs)} re-translations")

    reference = None
    for name in args.variants.split(","):
        engine = LlamaEngine(MT_DIR / "HY-MT1.5-1.8B-Q4_K_M.gguf", make_hy_mt_prompt(), args.threads,
                             **VARIANTS[name])
        engine.translate("Hello.", "en", "ja")
        t0 = time.perf_counter()
        outputs = [engine.translate(text, src, tgt).text for text, src, tgt in jobs]
        total = time.perf_counter() - t0
        engine.close()
        reference = reference or outputs
        same = sum(a == b for a, b in zip(outputs, reference))
        print(f"{name:22} total {total:6.1f}s  per call {1000 * total / len(jobs):5.0f} ms  "
              f"identical to baseline {same}/{len(jobs)}", flush=True)


if __name__ == "__main__":
    main()
