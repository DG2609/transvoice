"""Record what the PC is playing (system loopback, 16 kHz mono WAV) for a fixed time.

  python scripts/record_system.py .cache/clips/nhk.wav --seconds 240

Start it, then play the video; replay the file with scripts/replay.py.
"""
import argparse
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from transvoice.audio import Capture  # noqa: E402
from transvoice.paths import SAMPLE_RATE  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--seconds", type=float, required=True)
    ap.add_argument("--device", type=int)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    lock, level = threading.Lock(), [0.0]
    with sf.SoundFile(args.out, "w", samplerate=SAMPLE_RATE, channels=1) as f:
        def on_audio(x: np.ndarray) -> None:
            with lock:
                f.write(x)
                level[0] = max(level[0], float(np.abs(x).max()) if len(x) else 0.0)

        cap = Capture("system", on_audio, args.device)
        print("recording from", cap.start(), flush=True)
        end = time.time() + args.seconds
        while time.time() < end:
            time.sleep(5)
            with lock:
                print(f"{int(end - time.time()):4d}s left, peak {level[0]:.3f}", flush=True)
                level[0] = 0.0
        cap.stop()
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
