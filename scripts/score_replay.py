"""Score the final text of a replayed session against a reference transcript (CER for Japanese).

  python scripts/score_replay.py sessions/<session>.jsonl .cache/clips/ja_news.ref.txt --lang ja

Also reports how the sentences were cut: count, how many look like fragments (start with a particle or a
verb ending: "ったということです", "化しています"), and the lag.
"""
import argparse
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.metrics import asr_error_rate  # noqa: E402

# A Japanese sentence cannot start with these: it is the tail of the previous one.
_JA_FRAGMENT = re.compile(r"^(っ|ったと|たと|て|で|が|を|に|は|と|も|の|し|化|ます|まし|です|でし|超え)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("session", type=Path)
    ap.add_argument("ref", type=Path)
    ap.add_argument("--lang", default="ja")
    args = ap.parse_args()

    rows = [json.loads(line) for line in args.session.read_text(encoding="utf-8").splitlines()]
    rows.sort(key=lambda r: r["id"])
    mine = [r for r in rows if r["lang"] == args.lang]
    ref = "".join(re.sub(r"^\d+:\d+\s*", "", line) for line in args.ref.read_text(encoding="utf-8").splitlines()
                  if line and not line.startswith("#") and "(" not in line)
    hyp = "".join(r["text"] for r in mine)
    lags = sorted(r["timings"].get("lag_ms", 0) / 1000 for r in mine)
    frags = [r["text"] for r in mine if args.lang == "ja" and _JA_FRAGMENT.match(r["text"])]
    print(f"{args.session.name}: {len(mine)} {args.lang} sentences ({len(rows) - len(mine)} other), "
          f"CER {asr_error_rate([ref], [hyp], args.lang):.2f}%, fragments {len(frags)} {frags}, "
          f"lag p50 {statistics.median(lags):.1f} s p90 {lags[int(0.9 * len(lags))]:.1f} s")


if __name__ == "__main__":
    main()
