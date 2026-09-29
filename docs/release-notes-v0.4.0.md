TransVoice v0.4.0: checked on real recordings (a TV news clip and a street interview), sentence by sentence.

| TV news clip, whole pipeline, desktop CPU | v0.3.0 | v0.4.0 |
|---|---|---|
| Text error of the final subtitles (vs. YouTube captions) | 29.8% | **12.0%** |
| Sentences cut in half / invented sentences | 3 / 2 | **0 / 0** |
| Final translation after the speaker stops (median / p90) | 3.8 / 7.0 s | **2.5 / 3.4 s** |

| FLEURS, played in real time on a 4-core office-class CPU | v0.3.0 | v0.4.0 |
|---|---|---|
| Japanese speech recognition error | 9.4% | **6.6%** |
| Translation quality, COMET ja→vi / en→vi | 0.858 / 0.878 | **0.865 / 0.879** |
| Broken translations | 0% | 0% |

Fixes:
- The language of a first sentence or of a switch is re-checked on the next chunk (3 s of Japanese news had
  been taken for Vietnamese and a whole sentence was lost).
- Speech in another language (a Nepali interview inside Japanese news) is no longer turned into an invented,
  fluent translation: once the check on more audio confirms it, the sentence is dropped.
- Japanese is no longer cut inside words (the silence of っ/k/t); endings the speech recogniser invents at such
  cuts no longer end the sentence; periods it puts at every cut are ignored.
- A Japanese sentence ending followed by a short pause closes the sentence right away, even with background
  music (news sentences used to wait 3-5 s for the next chunk).
- Whole-sentence re-recognition covers sentences up to 16 s.
- On a slow PC under continuous speech, drafts pause so the final translations get the CPU.

Packages:

| System | File | How |
|---|---|---|
| Windows 10/11 x64 | `TransVoice-windows-x64.exe` | one file: double-click (models go next to it) |
| Windows 10/11 x64 | `TransVoice-windows-x64.zip` | folder: unzip, run `TransVoice.bat` |
| Debian / Ubuntu x64 | `transvoice_0.4.0_amd64.deb` | `sudo apt install ./transvoice_0.4.0_amd64.deb`, then `transvoice` |
| Linux x64 (glibc 2.27+) | `TransVoice-linux-x64.tar.gz` | unpack, `./run.sh` |
| macOS Apple Silicon | `TransVoice-macos-arm64.dmg` | added by GitHub Actions once Actions billing works again |

The first run downloads the models (~2.9 GB). Upgrading from v0.3.0: keep your `models/` folder next to the app.
Known limits: two people talking without a pause end up in one sentence; on a 4-core office PC continuous
dense speech (TV news) runs 12-16 s behind (conversation with pauses: 2-6 s).
