# TransVoice CPU benchmark

Measures local, CPU-only candidates for the JA/EN/VI translator: speech recognition (ASR) and
translation (MT). Scores accuracy, speed, CPU and RAM on this machine.

## Setup (Windows, Python 3.12)

```bash
python -m venv .venv && .venv/Scripts/python -m pip install sherpa-onnx ctranslate2 sentencepiece jiwer sacrebleu psutil numpy soundfile pyarrow pandas requests pytest
python -m venv .venv-comet && .venv-comet/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu && .venv-comet/Scripts/python -m pip install unbabel-comet
python scripts/download_models.py          # ~9 GB into models/
```

Data (not in git) lives in `data/`: FLEURS test splits for `ja_jp`, `en_us`, `vi_vn` and
`japanese-asr/ja_asr.reazonspeech_test`.

## Run

```bash
export PYTHONUTF8=1
.venv/Scripts/python scripts/run_asr.py                  # accuracy + timing, results/asr/
.venv/Scripts/python scripts/run_mt.py --n 100           # translations, results/mt/
.venv/Scripts/python scripts/run_mt.py --n 100 --source asr:sensevoice   # speech -> text -> translation
.venv-comet/Scripts/python scripts/run_comet.py          # COMET scores + "bad translation" rate
.venv/Scripts/python scripts/run_asr.py --speed-only --n 30 --n-reazon 60   # timing on an idle machine
.venv/Scripts/python scripts/run_mt.py --speed-only --n 20
.venv/Scripts/python scripts/report.py                   # results/REPORT.md
.venv/Scripts/python -m pytest
```

Each model runs in its own subprocess so RAM figures are per model. ASR uses 2 threads and MT uses
4 threads, matching the budget for an app that runs next to games and calls.
