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
| Translation | HY-MT1.5-1.8B IQ4_NL (imatrix, llama.cpp) + automatic glossary |
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

## Optimization round 2 (v0.2.0)

Measured with `scripts/run_live_eval.py`: 30 FLEURS utterances (15 ja, 15 en → vi) played in real time
through the engine on 4 E-cores (office-PC profile); final subtitles scored with CER/WER and COMET.

| | v0.1.0 | v0.2.0 |
|---|---|---|
| COMET en→vi / ja→vi | 0.804 / 0.843 | **0.880 / 0.857** |
| bad translations | up to 6.7% | **0%** |
| ASR error ja / en | 9.9% / 17.3% | **9.4% / 14.7%** |
| final lag p50 / p90 | 4.4 / 8.1 s | 6.2 / 9.1 s |

What changed and why:
- **False periods on English chunks.** SenseVoice ends every chunk with ".", even when the chunk was cut
  mid-sentence ("On August. 15"). That split every English sentence at each cut, so fragments were translated
  without context. Periods at forced cuts are now dropped, and at chunk joins only Japanese endings (real words)
  may close a sentence. English sentences per utterance went from 3.0 to 1.2 and COMET en→vi from 0.80 to 0.88.
- **Whole-sentence re-recognition** when a sentence ends (sentences ≤10 s whose audio is not shared with a
  neighbour, and only when ASR is not behind). Fixes words garbled at chunk joins (ja CER −20% relative).
  Unbounded, it pushed p90 lag to 15 s on office CPUs; the bounds keep it at 9 s.
- **Fast re-translation** (`mt_fast`): llama.cpp reuses the KV cache of the shared prompt prefix and drafts
  tokens with n-gram lookup. ~13–18% faster, COMET unchanged.
- **Stale drafts are cancelled** when a sentence ends (translations are streamed; closing the connection stops
  llama-server), so the final translation does not wait for an outdated draft.
- The final lag is higher than v0.1.0 because each final now covers a whole sentence instead of a fragment;
  drafts still appear while the sentence is being spoken.

## Optimization round 3: where the final lag goes

Per-stage medians on 4 E-cores (v0.2.0, `results/live/v3-stages.json`), measured from the end of the last chunk:
the pause is detected after 1.25 s, the final translation starts at 1.9 s (whole-sentence re-recognition, or a
stale draft being cancelled), and the translation itself takes 4.2 s. 39 of 40 finals needed a new translation
because the last draft never covered the finished text.

- **Streamed subtitles.** The translation is streamed into the overlay while the model writes it (never
  shrinking a draft already on screen). The first words of the final appear after 3.3 s instead of 6.2 s;
  quality and total lag unchanged (`v5-streaming-rerun`).
- **Rejected: re-recognising the sentence at every pause** (before it is known to be over): on 4 cores the
  extra ASR competes with translation; p50 6.9 s vs 6.2 s (`v4-early-rescore`).
- Live runs are sensitive to other load on the machine (one run on a busy PC: p50 10 s); `run_live_eval.py`
  now records the machine's average CPU load so such runs can be spotted.

The translation itself (4.2 s) is almost all token generation: a final re-translates a whole sentence,
~60 tokens at 13 tokens/s on 4 E-cores; prompt processing is only ~0.3 s thanks to prefix reuse
(`results/live/*.sentences.jsonl`, scratch benchmark of 30 FLEURS sentences: draft of 65% then final).

| final translation, 4 E-cores | ms | tok/s | output vs. no drafting |
|---|---|---|---|
| Q4_K_M, n-gram drafting off | 4873 | 13.1 | - |
| Q4_K_M, llama.cpp `ngram-mod` defaults (v0.2.0) | 4887 | 13.1 | 3 of 49 drafted tokens accepted |
| Q4_K_M, `ngram-mod` match 6, draft 1-16 | 4238 | 15.1 | 19/30 identical (near-tie word choices) |
| Q8_0 | 6461 | 9.6 | - |
| threads 3 instead of 4 | 5340 | 12.1 | - |
| imatrix Q4_0 | 3924 | 16.6 | - |
| imatrix IQ4_NL | 3846 | 16.8 | - |
| **imatrix IQ4_NL, `ngram-mod` match 6, draft 1-16** | **3545** | **18.3** | **30/30 identical** |

- **IQ4_NL instead of Q4_K_M.** Same quality on the 600-sentence text benchmark (COMET averaged over the 6
  directions: IQ4_NL 0.884, Q4_K_M 0.883, Q4_0 0.882; no real mistranslation in any of them) and 27% faster
  generation: llama.cpp repacks IQ4_NL/Q4_0 weights into CPU-friendly layouts. With those kernels, checking
  drafted tokens in a batch gives exactly the same output as generating one by one (with Q4_K_M it does not).
- **n-gram drafting tuned for sentences.** llama.cpp's `ngram-mod` defaults (24-token match, 48+ token drafts)
  target code edits and almost never fired. A 6-token match with drafts of up to 16 tokens reuses the previous
  draft's wording. Shorter matches (4) guess too often; rejected guesses cost CPU time.
- Q8_0 is slower (memory bound) and 3 threads only 8% slower than 4 (worth trying when the CPU is shared).

Live result (`v6-iq4nl`, 4 E-cores): final lag p50 / p90 **5.4 / 6.9 s** (v0.2.0: 6.2 / 9.5 s), final
translation 3.5 s (4.3 s); COMET en→vi 0.876 / ja→vi 0.858 (0.880 / 0.857, within noise at n=15), 0% bad.
Sentences closed at an ending 18, at a pause 19, too long 2, split 1.
Installs that already have Q4_K_M keep using it until `--download-models` fetches IQ4_NL.

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
