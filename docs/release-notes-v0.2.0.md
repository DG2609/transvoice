TransVoice v0.2.0: more accurate live translation. Measured on a 4-core office-class CPU (FLEURS speech played
in real time through the app, final subtitles scored):

| | v0.1.0 | v0.2.0 |
|---|---|---|
| Translation quality, COMET en→vi / ja→vi | 0.804 / 0.843 | **0.880 / 0.857** |
| Broken translations | up to 6.7% | **0%** |
| Speech recognition error ja / en | 9.9% / 17.3% | **9.4% / 14.7%** |
| Final translation after the speaker stops (median / p90) | 4.4 / 8.1 s | 6.2 / 9.1 s |

Changes:
- English sentences are no longer cut into fragments by false periods at chunk boundaries, so every sentence
  is translated with its full context.
- When a sentence ends, its whole audio is recognised again, fixing words garbled where chunks were cut.
- Faster re-translation (prompt-prefix reuse + n-gram speculative decoding in llama.cpp).
- Outdated draft translations are cancelled as soon as a sentence ends.
- Sentences are never closed while their chunks are still waiting to be processed (slow CPUs).
- Clear error message when llama-server cannot start (e.g. missing libgomp1 on minimal Linux).

## Downloads

| System | File |
|---|---|
| Windows 10/11 x64 | `TransVoice-windows-x64.zip` |
| Linux x64 (glibc 2.27+, e.g. Ubuntu 18.04+) | `TransVoice-linux-x64.tar.gz` |
| macOS Apple Silicon | added by GitHub Actions once Actions billing is enabled |

Models are unchanged from v0.1.0: an existing `models/` folder can be reused. New install: unpack, then run
`TransVoice --download-models` once (~2.9 GB).
