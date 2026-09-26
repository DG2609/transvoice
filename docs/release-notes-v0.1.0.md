TransVoice v0.1.0: first build. Live Japanese ⇄ English ⇄ Vietnamese speech translation, running only on
the CPU (no GPU, no internet after setup). Nothing leaves the machine.

## Downloads

| System | File | Status |
|---|---|---|
| Windows 10/11 x64 | `TransVoice-windows-x64.zip` | built and tested on Windows 11 |
| Linux x64 (glibc 2.27+, e.g. Ubuntu 18.04+) | `TransVoice-linux-x64.tar.gz` | built and tested in WSL Ubuntu (file translation) |
| macOS Apple Silicon | `TransVoice-macos-arm64.tar.gz` | will be added by the GitHub Actions build once Actions billing is enabled |

After unpacking, download the models once (~2.9 GB): `TransVoice --download-models`. Then run `TransVoice.bat`
(Windows) or `./run.sh` (macOS/Linux). See README.md for macOS (BlackHole) and Linux (`libportaudio2 libgomp1
pulseaudio-utils`) audio setup.

## What it does

- Captures system audio (calls, YouTube, meetings) and optionally the microphone.
- Live subtitles: the translation of the sentence being spoken updates every 2–5 s (grey) and is finalised
  when the sentence ends (white). Whole-sentence re-translation keeps the context instead of translating
  fragment by fragment.
- Detects Japanese / English / Vietnamese automatically, glossary support, session transcripts in `sessions/`.
- About 3.5–4 GB RAM. On a 4-core office-class CPU, final translations arrive ~3–5 s after a sentence ends.

## Models

parakeet-ja (JA speech), SenseVoice-Small (EN), Zipformer-30M-vi (VI), Whisper-base (language ID),
Silero VAD, HY-MT1.5-1.8B Q4_K_M via llama.cpp (translation). Benchmark: `results/REPORT.md`.
