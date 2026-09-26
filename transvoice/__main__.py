"""TransVoice: local Japanese/English/Vietnamese speech translation, CPU only.

  python -m transvoice                          # dịch âm thanh máy (cuộc gọi, video) sang tiếng Việt
  python -m transvoice --them ja --me vi        # cố định: họ nói tiếng Nhật
  python -m transvoice --sources system,mic     # dịch cả giọng của bạn (micro) sang ngôn ngữ của họ
  python -m transvoice --file meeting.wav       # dịch một file ghi âm, in kết quả ra màn hình
"""
import argparse
import json
import queue
import sys
import time
from pathlib import Path

import numpy as np

from .engine import LANGS, Engine, Settings
from .osutil import lower_own_priority
from .paths import MODELS, ROOT, SAMPLE_RATE
from .session import SessionLog


def load_glossary(path: Path | None) -> list[dict]:
    if not path:
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["terms"] if isinstance(data, dict) else data


def print_event(kind: str, payload, verbose: bool = False) -> None:
    if kind == "translated" and (payload.final or verbose):
        s, t = payload, payload.timings
        mark = "" if s.final else "~ "
        print(f"{mark}[{s.channel}·{s.lang}] {s.text}")
        if s.translation:
            print(f"    {mark}→ {s.target}: {s.translation}")
        print(f"    ({len(s.chunks)} mảnh, asr {t.get('asr_ms', 0)} ms, mt {t.get('mt_ms', 0)} ms, "
              f"trễ sau mảnh cuối {t.get('lag_ms', 0) / 1000:.1f} s, hàng chờ {t.get('backlog', 0)})", flush=True)
    elif kind in ("status", "error"):
        print(f"[{kind}] {payload}", flush=True)


def run_file(engine: Engine, path: Path, realtime: bool) -> None:
    import soundfile as sf

    from .audio import StreamResampler

    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = StreamResampler(sr)(audio)
    chunk = SAMPLE_RATE // 10
    for i in range(0, len(audio), chunk):
        engine.feed("them", audio[i : i + chunk])
        if realtime:
            time.sleep(0.1)
    engine.feed("them", np.zeros(SAMPLE_RATE, np.float32))  # trailing silence closes the last segment
    engine.flush()
    engine.wait_idle()


def main() -> None:
    ap = argparse.ArgumentParser(prog="transvoice", description="Dịch giọng nói Nhật/Anh/Việt ngay trên máy (CPU).")
    ap.add_argument("--me", default="vi", choices=LANGS, help="ngôn ngữ của bạn; lời của họ được dịch sang đây")
    ap.add_argument("--them", default="auto", choices=("auto", *LANGS), help="ngôn ngữ của họ (auto = tự nhận)")
    ap.add_argument("--sources", default="system", help="system, mic, hoặc system,mic")
    ap.add_argument("--glossary", type=Path, help="file JSON thuật ngữ, xem glossary.example.json")
    ap.add_argument("--file", type=Path, help="dịch file âm thanh thay vì nghe trực tiếp")
    ap.add_argument("--realtime", action="store_true", help="với --file: đưa âm thanh vào theo tốc độ thật")
    ap.add_argument("--no-overlay", action="store_true", help="không hiện cửa sổ phụ đề, chỉ in ra console")
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--download-models", action="store_true", help="tải model (~2,9 GB) vào thư mục models/")
    ap.add_argument("--system-device", type=int, help="số thiết bị 'system' trong --list-devices (mặc định: loa mặc định)")
    ap.add_argument("--mic-device", type=int, help="số thiết bị 'mic' trong --list-devices (mặc định: micro mặc định)")
    ap.add_argument("--asr-threads", type=int, default=2)
    ap.add_argument("--mt-threads", type=int, default=4)
    ap.add_argument("--sessions", type=Path, default=ROOT / "sessions", help="thư mục lưu lịch sử")
    ap.add_argument("--no-drafts", action="store_true", help="chỉ dịch khi hết câu (nhẹ CPU hơn, kém 'trực tiếp')")
    ap.add_argument("--verbose", action="store_true", help="in cả bản dịch nháp ra console")
    ap.add_argument("--record", type=Path, help="ghi lại âm thanh máy đã nghe (16 kHz WAV) để kiểm tra lại bằng --file")
    args = ap.parse_args()

    if sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    if args.list_devices:
        from .audio import describe_devices

        print("\n".join(describe_devices()))
        return

    if args.download_models:
        from .download import APP_MODELS, download

        download(APP_MODELS)
        print(f"Xong. Model nằm trong {MODELS}")
        return

    missing = [p for p in (MODELS / "mt" / "HY-MT1.5-1.8B-Q4_K_M.gguf", MODELS / "vad" / "silero_vad.onnx",
                           MODELS / "bin" / "llama.cpp") if not p.exists()]
    if missing:
        print(f"Chưa có model trong {MODELS}.\nChạy trước:  TransVoice.exe --download-models  (khoảng 2,9 GB)")
        sys.exit(1)
    lower_own_priority()

    settings = Settings(my_lang=args.me, their_lang=args.them, glossary=load_glossary(args.glossary),
                        asr_threads=args.asr_threads, mt_threads=args.mt_threads, drafts=not args.no_drafts)
    log = SessionLog(args.sessions)
    ui_q: queue.Queue = queue.Queue()
    use_overlay = not args.no_overlay and not args.file

    def on_event(kind: str, payload) -> None:
        if kind == "translated" and payload.final:
            log.write(payload)
        print_event(kind, payload, args.verbose)
        if use_overlay:
            ui_q.put((kind, payload))

    engine = Engine(settings, on_event)
    engine.start()
    try:
        if args.file:
            run_file(engine, args.file, args.realtime)
            print(f"\nĐã lưu: {log.md}")
            return

        from .audio import Capture

        channel_of = {"system": "them", "mic": "me"}
        device_of = {"system": args.system_device, "mic": args.mic_device}
        recorder = None
        if args.record:
            import soundfile as sf

            recorder = sf.SoundFile(args.record, "w", samplerate=SAMPLE_RATE, channels=1)

        def on_audio(channel: str, x: np.ndarray) -> None:
            if recorder is not None and channel == "them":
                recorder.write(x)
            engine.feed(channel, x)

        captures = []
        for src in args.sources.split(","):
            cap = Capture(src, lambda x, ch=channel_of[src]: on_audio(ch, x), device_of[src])
            try:
                name = cap.start()
            except RuntimeError as e:  # e.g. no loopback driver on macOS: explain instead of a traceback
                print(f"[error] {e}")
                sys.exit(2)
            captures.append(cap)
            on_event("status", f"Nghe {'âm thanh máy' if src == 'system' else 'micro'}: {name}")
        on_event("status", f"Đang nghe · dịch sang {args.me.upper()}")
        try:
            if use_overlay:
                from .overlay import Overlay

                Overlay(ui_q, on_close=lambda: None).run()
            else:
                while True:
                    time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            for cap in captures:
                cap.stop()
            if recorder is not None:
                recorder.close()
    finally:
        engine.stop()
        print(f"Lịch sử phiên: {log.md}")


if __name__ == "__main__":
    main()
