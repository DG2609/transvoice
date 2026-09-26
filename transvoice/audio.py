"""Microphone and system-audio capture on Windows, macOS and Linux, delivered as 16 kHz mono float32."""
import subprocess
import threading
from typing import Callable

import numpy as np

from .osutil import IS_LINUX, IS_WIN
from .paths import SAMPLE_RATE


class StreamResampler:
    """Anti-aliased resampling to 16 kHz that keeps filter state across chunks."""

    def __init__(self, sr_in: int, sr_out: int = SAMPLE_RATE, taps: int = 63):
        self.step = sr_in / sr_out
        cutoff = min(0.5, 0.5 * sr_out / sr_in) * 0.9  # fraction of sr_in, a bit below the output Nyquist
        n = np.arange(taps) - (taps - 1) / 2
        h = np.sinc(2 * cutoff * n) * np.hamming(taps)
        self.h = (h / h.sum()).astype(np.float32)
        self.hist = np.zeros(taps - 1, np.float32)
        self.prev = np.zeros(1, np.float32)  # last filtered sample of the previous chunk
        self.pos = 1.0  # next output position, indexed into [prev, *filtered chunk]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.step == 1:
            return x.astype(np.float32)
        buf = np.concatenate([self.hist, x.astype(np.float32)])
        self.hist = buf[-(len(self.h) - 1):]
        y = np.concatenate([self.prev, np.convolve(buf, self.h, mode="valid")])
        self.prev = y[-1:]
        # Positions in the last interval wait for the next chunk, which starts with this chunk's last sample.
        idx = np.arange(self.pos, len(y) - 1, self.step)
        self.pos = (idx[-1] + self.step if len(idx) else self.pos) - (len(y) - 1)
        return np.interp(idx, np.arange(len(y)), y).astype(np.float32)


class AutoGain:
    """Slow automatic gain control. Quiet calls/videos (peaks around -35 dBFS) are otherwise invisible to
    Silero VAD, which then drops whole sentences. Never attenuates; boosts by at most `max_gain`."""

    def __init__(self, target_peak: float = 0.5, max_gain: float = 30.0, release_s: float = 3.0,
                 sample_rate: int = SAMPLE_RATE):
        self.target, self.max_gain = target_peak, max_gain
        self.release_per_sample = np.exp(np.log(0.1) / (release_s * sample_rate))  # envelope falls 10x per release_s
        self.env = 0.0
        self.gain = 1.0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if len(x) == 0:
            return x
        peak = float(np.abs(x).max())
        # Fast attack, slow release, so the gain follows the speaker's level rather than each syllable.
        self.env = max(peak, self.env * self.release_per_sample ** len(x))
        new_gain = float(np.clip(self.target / max(self.env, 1e-6), 1.0, self.max_gain))
        ramp = np.linspace(self.gain, new_gain, len(x), dtype=np.float32)  # no clicks between chunks
        self.gain = new_gain
        return np.clip(x * ramp, -1.0, 1.0).astype(np.float32)


MAC_LOOPBACK_NAMES = ("BlackHole", "Loopback", "Soundflower")
MAC_HELP = ("macOS does not let apps record the system output directly. Install the free BlackHole 2ch driver "
            "(https://existential.audio/blackhole/), create a Multi-Output Device with your speakers + BlackHole in "
            "Audio MIDI Setup, select it as output, then start TransVoice again (or pass --system-device).")


def describe_devices() -> list[str]:
    if IS_WIN:
        return _describe_wasapi()
    import sounddevice as sd

    lines = [f"{i:>3}  {'mic':6}  {d['name']}" for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0]
    if IS_LINUX:
        lines.append("  -  system  default output monitor (PulseAudio/PipeWire, via parec)")
    return lines


class Capture:
    """Captures one device and hands 16 kHz mono chunks to `on_audio` from a background thread (keep it fast).

    Windows: WASAPI (mic, and loopback of the speakers for "system").
    Linux:   PortAudio for the mic; "system" records the default output's monitor with parec (PulseAudio/PipeWire).
    macOS:   PortAudio; "system" needs a loopback driver such as BlackHole (see MAC_HELP).
    """

    def __init__(self, kind: str, on_audio: Callable[[np.ndarray], None], device_index: int | None = None):
        assert kind in ("mic", "system")
        self.kind, self.on_audio, self.device_index = kind, on_audio, device_index
        self._stop: Callable[[], None] = lambda: None

    def start(self) -> str:
        if IS_WIN:
            return self._start_wasapi()
        if self.kind == "system" and IS_LINUX and self.device_index is None:
            return self._start_parec()
        return self._start_portaudio()

    def stop(self) -> None:
        self._stop()

    # ---- Windows ----
    def _start_wasapi(self) -> str:
        import pyaudiowpatch as pa

        p = pa.PyAudio()
        if self.device_index is not None:
            info = p.get_device_info_by_index(self.device_index)
        elif self.kind == "system":
            info = p.get_default_wasapi_loopback()
        else:
            wasapi = p.get_host_api_info_by_type(pa.paWASAPI)
            info = p.get_device_info_by_index(wasapi["defaultInputDevice"])
        channels = max(1, min(2, int(info["maxInputChannels"])))
        rate = int(info["defaultSampleRate"])
        resample = StreamResampler(rate)

        def callback(in_data, frame_count, time_info, status):
            x = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
            if channels > 1:
                x = x.reshape(-1, channels).mean(axis=1)
            self.on_audio(resample(x))
            return (None, pa.paContinue)

        stream = p.open(format=pa.paInt16, channels=channels, rate=rate, input=True,
                        input_device_index=info["index"], frames_per_buffer=rate // 10, stream_callback=callback)

        def stop() -> None:
            stream.stop_stream()
            stream.close()
            p.terminate()

        self._stop = stop
        return info["name"]

    # ---- macOS / Linux ----
    def _start_portaudio(self) -> str:
        import sounddevice as sd

        index = self.device_index
        if index is None and self.kind == "mic":
            index = sd.default.device[0]
        elif index is None:  # macOS system audio
            index = next((i for i, d in enumerate(sd.query_devices())
                          if d["max_input_channels"] > 0 and any(n in d["name"] for n in MAC_LOOPBACK_NAMES)), None)
            if index is None:
                raise RuntimeError(MAC_HELP)
        info = sd.query_devices(index)
        channels = max(1, min(2, int(info["max_input_channels"])))
        rate = int(info["default_samplerate"])
        resample = StreamResampler(rate)

        def callback(indata, frames, time_info, status):
            x = indata.mean(axis=1) if channels > 1 else indata[:, 0]
            self.on_audio(resample(np.ascontiguousarray(x, dtype=np.float32)))

        stream = sd.InputStream(device=index, channels=channels, samplerate=rate, dtype="float32",
                                blocksize=rate // 10, callback=callback)
        stream.start()
        self._stop = lambda: (stream.stop(), stream.close())
        return info["name"]

    def _start_parec(self) -> str:
        try:
            sink = subprocess.run(["pactl", "get-default-sink"], capture_output=True, text=True,
                                  check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as e:
            raise RuntimeError("System audio on Linux needs PulseAudio or PipeWire with pactl/parec "
                               "(package pulseaudio-utils).") from e
        source = f"{sink}.monitor"
        proc = subprocess.Popen(["parec", f"--device={source}", "--format=s16le", f"--rate={SAMPLE_RATE}",
                                 "--channels=1", "--latency-msec=100"], stdout=subprocess.PIPE)

        def pump() -> None:
            chunk = SAMPLE_RATE // 10 * 2  # 100 ms of 16-bit samples
            while True:
                data = proc.stdout.read(chunk)
                if not data:
                    return
                self.on_audio(np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0)

        threading.Thread(target=pump, daemon=True).start()
        self._stop = lambda: (proc.terminate(), proc.wait(5))
        return source


def _describe_wasapi() -> list[str]:
    import pyaudiowpatch as pa

    p = pa.PyAudio()
    try:
        wasapi = p.get_host_api_info_by_type(pa.paWASAPI)
        lines = []
        for i in range(wasapi["deviceCount"]):
            d = p.get_device_info_by_host_api_device_index(wasapi["index"], i)
            if d["maxInputChannels"] > 0:
                kind = "system" if d.get("isLoopbackDevice") else "mic"
                lines.append(f"{d['index']:>3}  {kind:6}  {d['name']}")
        return lines
    finally:
        p.terminate()
