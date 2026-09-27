TransVoice v0.3.0: subtitles that appear sooner, same translation quality. Measured on a 4-core office-class
CPU (FLEURS speech played in real time through the app, final subtitles scored):

| | v0.2.0 | v0.3.0 |
|---|---|---|
| Final translation after the speaker stops (median / p90) | 6.2 / 9.5 s | **4.4 / 6.9 s** |
| First words of the final translation on screen (median) | 6.2 s | **3.0 s** |
| Translation quality, COMET en→vi / ja→vi | 0.880 / 0.857 | 0.878 / 0.858 |
| Broken translations | 0% | 0% |

Changes:
- Translations are streamed into the subtitles word by word while they are generated (a draft already on
  screen is never replaced by a shorter one).
- New translation weights: HY-MT1.5-1.8B IQ4_NL. Same quality as Q4_K_M on 600 test sentences in all six
  JA/EN/VI directions and 27% faster on CPUs.
- Speculative decoding tuned for re-translating a growing sentence: the previous draft's wording is reused
  and checked in batches (output identical to normal decoding).
- The whole-sentence second recognition pass now runs for Japanese (where it prevents mistranslations) and
  not for English (where it only delayed the result by 1.5 s).
- In continuous speech, a finished sentence's final translation no longer waits behind the draft of the
  sentence that follows (replay of a real Japanese interview on 4 cores: p90 lag 12.0 → 9.9 s).
- A pause right after a word that cannot end a sentence ("…at the", "…雨が", "…của") no longer ends the
  sentence unless it lasts 1.2 s.

## Downloads

| System | File |
|---|---|
| Windows 10/11 x64 | `TransVoice-windows-x64.zip` |
| Linux x64 (glibc 2.27+, e.g. Ubuntu 18.04+) | `TransVoice-linux-x64.tar.gz` |
| macOS Apple Silicon | added by GitHub Actions once Actions billing is enabled |

Upgrading: keep your `models/` folder and run `TransVoice --download-models` once; it only fetches the new
translation weights (1.1 GB). Without it the app keeps working with the old Q4_K_M weights, just slower.
New install: unpack, then run `TransVoice --download-models` once (~2.9 GB).
