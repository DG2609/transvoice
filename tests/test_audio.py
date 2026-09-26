import numpy as np

from transvoice.audio import AutoGain, StreamResampler


def _tone(freq, sr, seconds):
    t = np.arange(int(sr * seconds)) / sr
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def test_resample_48k_keeps_speech_band_tone():
    out = StreamResampler(48000)(_tone(1000, 48000, 1.0))
    assert abs(len(out) - 16000) <= 1
    spectrum = np.abs(np.fft.rfft(out[1000:-1000]))
    peak_hz = np.argmax(spectrum) * 16000 / len(out[1000:-1000])
    assert abs(peak_hz - 1000) < 5
    assert 0.9 < np.abs(out[1000:-1000]).max() < 1.05


def test_resample_filters_out_of_band_tone():
    out = StreamResampler(48000)(_tone(12000, 48000, 1.0))  # above the 8 kHz output Nyquist
    assert np.abs(out[1000:-1000]).max() < 0.05


def test_chunked_equals_one_shot():
    x = _tone(440, 44100, 1.0)
    one = StreamResampler(44100)(x)
    r = StreamResampler(44100)
    chunked = np.concatenate([r(c) for c in np.array_split(x, 17)])
    n = min(len(one), len(chunked))
    assert abs(len(one) - len(chunked)) <= 1
    assert np.allclose(one[:n], chunked[:n], atol=1e-4)


def _run_agc(agc, x, chunk=1600):
    return np.concatenate([agc(x[i : i + chunk]) for i in range(0, len(x), chunk)])


def test_agc_boosts_quiet_speech_level():
    out = _run_agc(AutoGain(), 0.017 * _tone(300, 16000, 2.0))
    assert 0.4 < np.abs(out[16000:]).max() <= 0.51  # -35 dBFS peaks brought to about -6 dBFS


def test_agc_never_attenuates_loud_audio():
    x = 0.9 * _tone(300, 16000, 1.0)
    assert np.allclose(_run_agc(AutoGain(), x), x, atol=1e-6)


def test_agc_gain_is_capped_on_silence():
    out = _run_agc(AutoGain(), 1e-4 * _tone(300, 16000, 5.0))
    assert np.abs(out).max() <= 30 * 1e-4 + 1e-6


def test_16k_passthrough():
    x = _tone(440, 16000, 0.1)
    assert np.array_equal(StreamResampler(16000)(x), x)
