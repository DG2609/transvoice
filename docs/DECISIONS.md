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

Per-sentence timings (`v6-iq4nl.sentences.jsonl`) then showed the whole-sentence re-recognition changing the
text of 19 of 20 sentences and taking 0.5-1.4 s, during which the translator waits. Measured per language:

| live, 4 E-cores | ja→vi COMET | en→vi COMET | ja lag p50 | en lag p50 |
|---|---|---|---|---|
| re-recognise ja + en (`v6-iq4nl`) | 0.858 | 0.876 | 5.3 s | 5.5 s |
| never (`v7-norescore`) | 0.833, 6.7% bad | 0.878 | 4.2 s | 4.1 s |
| **ja only (`v8-rescore-ja`)** | **0.858, 0% bad** | **0.878** | 5.3 s | **3.9 s** |

- **Re-recognition only for Japanese (and Vietnamese, whose 30M zipformer costs almost nothing).** English WER
  is better with it (14.7% vs 17.3%) but the translation is not: HY-MT copes with the small errors.
  Overall final lag p50 / p90: **4.4 / 6.9 s**.
- **Rejected: a shorter pause (0.5 s instead of 0.8 s).** Sentences closed 0.27 s earlier but the lag did not
  move (the translator was still busy), English sentences were split more often (1.33 vs 1.2 per utterance)
  and en→vi COMET dropped to 0.864.
- **A pause right after a word that cannot end a sentence** ("…at the", "…雨が", "…của") must last 1.2 s
  before the sentence is closed: the speaker is looking for a word. (FLEURS has no such hesitations, so
  this is a unit-tested rule, not a measured one.)
- Rejected: 4 ASR threads instead of 2 (re-recognition 580 vs 710 ms, but p90 lag 7.6 vs 6.9 s because
  recognition then competes with translation).

### Real conversation (replay of a Japanese street interview, `.cache/youtube_run2.wav`, 4 E-cores)

FLEURS is read speech with pauses between sentences; a street interview is continuous, with fillers and run-on
sentences. `scripts/replay.py` plays a recording through the app in real time with setting overrides and
prints the translator's time split (drafts / finals / cancelled work).

| 27 sentences | lag p50 | p90 | max |
|---|---|---|---|
| v0.3.0 as tagged | 6.1 s | 12.0 s | 13.3 s |
| **+ a final cancels a later sentence's draft** | 6.1 s | **9.9 s** | **11.3 s** |
| sentence cap 8 s instead of 12 s (rejected; FLEURS en→vi COMET 0.873 vs 0.878) | 6.4 s | 9.7 s | 12.1 s |
| 3 translation threads instead of 4 (no clear change) | 6.1 s | 9.4 s | 12.0 s |

- **A final translation cancels a running draft of a later sentence.** In continuous speech the next sentence
  starts while the finished one is being re-recognised; the translator began the new sentence's draft and the
  final then waited 3-4 s behind it.
- The translator is busy only ~25% of this recording (drafts 72 s, finals 82 s, cancelled 25 s of 704 s); it
  saturates only during long runs of speech.
- The largest remaining delay: a Japanese sentence that ends in a polite form exactly at a chunk cut
  ("…ですね" | next words) is only closed when the next chunk has been recognised (3-6 s later), because
  "…ですけど", "…ますが" would continue it. Its full translation is already on screen as a grey draft.

## Round 4: real recordings ("slow and wrong")

A 4-minute TBS news clip (`.cache/clips/ja_news.wav`, recorded with `scripts/record_system.py`, reference =
YouTube's auto captions, which have their own errors) and the street interview were replayed through the app
and read sentence by sentence. `scripts/compare_asr_clip.py` compares ASR models on the app's own chunks,
`scripts/score_replay.py` scores a replayed session.

What was wrong, by cause:
- **Language ID locked a whole sentence into the wrong language**: the first 3.1 s of news were heard as
  Vietnamese and, being longer than 2 s, trusted; 14 s were lost. Now a first sentence or a switch of language is
  confirmed on the next chunk (every later chunk, alone or together, was Japanese).
- **Speech in another language became fluent invented Vietnamese**: a Nepali interview inside the news was
  recognised as Japanese/English nonsense and translated. Whisper does say "ne"/"tl" there, but that answer
  used to *confirm* the Japanese label. Now a sentence clearly in another language (not zh/yue/ko, which
  Whisper confuses with Japanese) is dropped and its draft line removed.
- **False periods from the Japanese ASR**: parakeet ends every chunk with "。", even "けがをしていて警察は。";
  the sentence was split there and the translator invented the missing half. Periods at forced cuts are now
  dropped for every language (polite endings like です/ました still end a sentence).
- **Cuts inside Japanese words**: the silence of っ, /k/, /t/ closures (60-150 ms) passed the 64 ms "gap
  between words" test, so words were cut in two ("電話をかけ|かけ", "出血性|性ショック") and the ASR hallucinated
  endings at the cut ("1400人を超え" -> "超えています", "作っ|たり" -> "作っています"), which then ended the
  sentence. reazonspeech-k2 has the same problem at cuts (same CER on the clip), so the ASR model is not the
  fix. Three changes, chosen with `scripts/sweep_cuts.py` (chunk-level CER) and whole-pipeline runs:
  - Japanese needs 192 ms of quiet to cut, relaxed towards 64 ms as the chunk nears the 5 s hard limit (a
    fixed 192 ms left read speech without a place to cut: hard cuts mid-word lost "安全に泳ぐこと" once).
    English keeps 64 ms (longer English chunks get false periods inside: FLEURS en→vi 0.872 vs 0.878).
  - A Japanese ending at a cut ends the sentence only if the cut was a real pause (>= 192 ms); at a shorter
    gap it is probably invented.
  - Whole-sentence re-recognition up to 16 s instead of 10 s (only Japanese is re-recognised now); the longer,
    unsplit sentences otherwise kept every garbled join.
- **Slow closes in continuous speech**: news sentences end at a cut ("…ました") and waited 3-5 s for the next
  chunk; background music keeps the VAD from reporting the pause. The pause is now measured against the voice
  level; a Japanese sentence ending followed by 0.35 s of quiet closes the sentence.

| TBS news clip, whole pipeline, full CPU | text error | fragments | invented sentences | lag p50 / p90 |
|---|---|---|---|---|
| before this round | 29.8% | 3 | 2 | 3.8 / 7.0 s |
| + language re-check, false periods, pause after an ending | 20.1% | 3 | 2 | 2.4 / 5.0 s |
| + another language is dropped | 16.5% | 3 | 0 | 3.0 / 5.0 s |
| + 192 ms cuts for Japanese (fixed) | 11.6% | 1 | 0 | 2.9 / 5.1 s |
| **final: 192 ms relaxing to 64 ms, endings trusted only at pauses, re-recognition up to 16 s** | **12.0%** | **0** | **0** | 2.5 / 3.3 s |

| FLEURS live (30 utterances, 4 E-cores) | ja CER | COMET ja→vi | COMET en→vi | lag p50 / p90 |
|---|---|---|---|---|
| v0.3.0 | 9.4% | 0.858 | 0.878 | 4.6 / 7.0 s |
| fixed 192 ms (`v13-all`) | 9.6% | 0.854 | 0.880 | 4.9 / 7.1 s |
| relaxing cuts + trusted endings (`v15-ramp-trust`) | 8.2% | 0.863 | 0.879 | 4.8 / 7.3 s |
| **+ re-recognition up to 16 s (`v17-load-guard`, final)** | **6.6%** | **0.865** | 0.879 | 4.9 / 8.8 s |

Dense news on 4 E-cores stays 12-16 s behind whatever the settings (spinning threads off, 3 translation
threads, 1 ASR thread: no change): the final translations alone keep the translator ~60% busy and are long
(110 characters of Japanese -> 336 of Vietnamese in 15 s). The same clip on the desktop: 2.5 / 3.3 s. Drafts
pause while the translator is over 85% busy; that halves the wasted draft work but does not change the lag.

Translation errors that remain are the model's own (HY-MT1.5-1.8B): "アルバイト" -> "tên trộm",
"頭を切りつけられる" -> "bị cắt đầu", "夫に腕をつかまれた" with the roles reversed, "手塚さん" given a male title.
Tried and rejected, measured:
- **Context from earlier sentences** (BSD business dialogues, JA→EN, 315 sentences, COMET): none 0.812;
  HY-MT's own contextual prompt 0.769 (1 sentence) / 0.741 (2) — the 1.8B model translates the context too
  ("え、私が？" -> "Oh, I… Mr. Mori, it's a phone call in English…"); earlier sentences as chat history
  0.809 / 0.810. Kept in `mt.py` and `scripts/run_context_mt.py` for re-testing with another model.
- **Translating through English** (ja→en→vi): fixed 3 of the 6 news errors above but not on average
  (FLEURS ja→vi COMET 0.868 vs 0.869 direct; vi→ja 0.895 vs 0.889 with 4% vs 1% bad) at twice the time.

### Tried and removed: a new sentence at a change of speaker

Interviewer and interviewee talking without a pause end up in one sentence. CAM++ speaker embeddings
(3D-Speaker, 28 MB, 35 ms per chunk; `scripts/eval_speaker_change.py`) separate FLEURS speakers well: at
cosine 0.2 on chunks >= 1.5 s, 1.9% of same-speaker joins would split while 84% of changes to the other
gender are caught (the WeSpeaker CAM++ model could not tell speakers apart at all). In the pipeline FLEURS was
unchanged (COMET ja→vi 0.865), but on the street interview it split once correctly, once in the middle of one
person's sentence, and the chunk after a split, re-identified alone in street noise, was taken for another
language and lost. Removed; the model stays in the download manifest for the evaluation script.

That also showed that dropping "another language" on one chunk's language ID was too hasty: now one guess only
makes the sentence unconfirmed, and it is dropped when the check on more audio agrees (the Nepali interview in
the news is still dropped: Whisper says "tl", then "ne").

## Known limitations

- On a 4-core office PC, continuous dense speech (TV news) runs 12-16 s behind; conversation with pauses 2-6 s.
- Two people talking without a pause end up in one sentence (question and answer); there is no speaker
  separation.

- A one-chunk sentence shorter than 2 s right after a language switch can still be recognised in the previous
  language (language ID is unreliable on so little audio).
- HY-MT-1.8B still errs on idioms, names and long ASR-garbled sentences (わびさび, ドラえもん, a negation inside a
  run-on sentence). A 4B model fixes some of these but is ~2.5x slower on office CPUs.
- Per-application capture (only Zoom, only the browser) is not implemented yet; the whole system output is captured.
- Tested on FLEURS, a TV news clip and a street interview; still needs real call/meeting recordings.

## Must-haves found during testing

- Lowercase ALL-CAPS ASR output (Vietnamese zipformer) before translating.
- Run llama-server with `--cache-ram 0 --ctx-checkpoints 0` so RAM stays flat.
- Glossary for Vietnamese weekdays ("thứ năm" = Thursday) plus user-defined terms.
- Drop ASR output from non-speech (VAD gate; ignore 1–2 character fillers).
