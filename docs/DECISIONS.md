# TransVoice — decisions

## Scope

- **Phase 1: text only.** Speech → recognized text → translated text shown as subtitles. This is the priority.
- **Later: TTS / voice-over.** Deferred. Candidates from research (not benchmarked yet): Supertonic 3
  (JA/EN/VI, repo archived 2026-09-09), piper-plus (JA), Kokoro (EN), VieNeu-TTS Nano (VI).

## Constraints

- Runs on company PCs with **no GPU**: CPU only.
- Translation accuracy first; RAM around 4 GB is the target, a bit more is acceptable if the app stays stable.
- Languages: Japanese, English, Vietnamese in every direction.

## Chosen stack (from the benchmark, see `results/REPORT.md`)

| Stage | Model |
|---|---|
| ASR Japanese | parakeet-tdt_ctc-0.6b-ja (sherpa-onnx, int8) |
| ASR English | SenseVoice-Small 2024-07-17 (sherpa-onnx, int8) |
| ASR Vietnamese | Zipformer-30M-vi (sherpa-onnx, int8) |
| Language ID | Whisper-base (sherpa-onnx) |
| Translation | HY-MT1.5-1.8B Q4_K_M (llama.cpp) + automatic glossary |
| Summary / Q&A (on demand, later) | Qwen3.5-4B Q4_K_M |

## App prototype (2026-09-26)

- **Python prototype first** (`transvoice/`), reusing the benchmarked code. Rust/MSVC are not installed and
  Tauri needs both; the models dominate RAM (~3.5 GB), the Python shell adds ~100 MB. Port to Tauri/Rust later
  if installer size or startup time matters.
- Audio: PyAudioWPatch (WASAPI) for mic and system loopback, resampled to 16 kHz in-process.
- Automatic gain control before VAD: quiet calls/recordings (peaks around -35 dBFS) made Silero VAD drop
  whole sentences (20 s of 85 s in the test stream). With AGC every sentence is captured.
- Overlay: Tkinter, borderless, topmost, click-through toggle via Win32 styles, global hotkeys via RegisterHotKey.

## Live subtitles that keep context (2026-09-26)

Translating each chunk on its own ("dịch tới đâu ra tới đó") loses context: tested on real chunks, HY-MT
turned "明日の会議には" into "Người tham dự cuộc họp ngày mai là…" and invented content for dangling English
clauses. Telling HY-MT to "translate only the continuation" does not work (it re-translates the context).

Chosen design, re-translation:
- VAD chunks of 2.5–5 s, cut in a pause between words (≥64 ms clearly below the speech level).
- Chunks append to the channel's open **sentence**; the whole sentence-so-far is re-translated after every
  chunk, so the subtitle grows live (grey) and the model always sees the full sentence (Japanese verb and
  negation come last). The last translation after the sentence ends is final (white) and is the only one saved.
- If the translator is busy, stale drafts are skipped; only the newest text is translated (keeps up on slow CPUs).
  `--no-drafts` translates finished sentences only.
- Sentence ends: explicit punctuation, Japanese polite endings followed by a new sentence (ですか, ますよね,
  ました… but not "ましたが" or the filler "ですね、"), a real pause (VAD silent ≥0.8 s), or ~12–15 s of audio.
  Mixing a question and its answer in one sentence made the model swap "bạn"/"tôi"; splitting fixed it.
- Short first chunks after a language switch are re-checked with language ID once the sentence has ≥2 s of
  audio, and earlier chunks are re-recognised with the right model.

Replay of a real YouTube street-interview recording (desktop CPU): sentences median 4.3 s, drafts every chunk,
final translation median 3.0 s after the speaker's last chunk. HY-MT Q8_0 was tried against Q4_K_M on the
mistranslated sentences: same errors (they come from long, ASR-garbled input), 50% slower, +0.8 GB → keep Q4.

## Known limitations

- A one-chunk sentence shorter than 2 s right after a language switch can still be recognised in the previous
  language (language ID is unreliable on so little audio).
- HY-MT-1.8B still errs on idioms, names and long ASR-garbled sentences (わびさび, ドラえもん, a negation inside a
  run-on sentence). A 4B model fixes some of these but is ~2.5x slower on office CPUs.
- Per-application capture (only Zoom, only the browser) is not implemented yet; the whole system output is captured.
- Tested on FLEURS and synthetic playback; still needs real call/meeting recordings.

## Must-haves found during testing

- Lowercase ALL-CAPS ASR output (Vietnamese zipformer) before translating.
- Run llama-server with `--cache-ram 0 --ctx-checkpoints 0` so RAM stays flat.
- Glossary for Vietnamese weekdays ("thứ năm" = Thursday) plus user-defined terms.
- Drop ASR output from non-speech (VAD gate; ignore 1–2 character fillers).
