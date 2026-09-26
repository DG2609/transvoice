"""Build TransVoice for the current OS with PyInstaller.

  python packaging/build.py            # runs the tests, then builds
Output: dist/TransVoice/ and dist/TransVoice-<os>-<arch>.zip (Windows) or .tar.gz (macOS/Linux).
Models are not bundled: run "TransVoice --download-models" once per machine, or copy a models/ folder
next to the executable.
"""
import argparse
import platform
import shutil
import stat
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
IS_WIN, IS_MAC = sys.platform == "win32", sys.platform == "darwin"

EXCLUDE = ["pandas", "pyarrow", "ctranslate2", "sentencepiece", "sacrebleu", "jiwer", "PIL", "pytest"]


# PyInstaller copies these from the build machine. Every desktop distro ships them (libportaudio2 is listed
# in the README), and bundling the build machine's copies would demand its glibc version on every user's PC.
LINUX_SYSTEM_LIBS = ("libstdc++.so", "libgcc_s.so", "libmvec.so", "libasound.so", "libsystemd.so", "libX", "libxcb",
                     "libportaudio.so", "libjack.so", "libpulse", "libsndfile.so", "libdbus", "libcap.so", "libgcrypt",
                     "liblzma.so", "libzstd.so", "liblz4.so", "libgpg-error", "libFLAC.so", "libapparmor.so",
                     "libasyncns.so", "libdb-", "libmp3lame.so", "libmpg123.so", "libogg.so", "libopus.so",
                     "libvorbis", "libz.so")
# Libraries repaired into wheels by auditwheel carry a hash in their name (libasound-fb5348bf.so) and were
# built for old glibc, so the plain-name prefixes above never match them.


def drop_linux_system_libs(app: Path) -> None:
    internal = app / "_internal"
    for f in internal.rglob("*.so*"):
        if f.name.startswith(LINUX_SYSTEM_LIBS) and "_soundfile_data" not in f.parts and "sherpa_onnx" not in f.parts:
            f.unlink()


def platform_tag() -> str:
    os_name = "windows" if IS_WIN else "macos" if IS_MAC else "linux"
    arch = {"amd64": "x64", "x86_64": "x64", "arm64": "arm64", "aarch64": "arm64"}.get(platform.machine().lower(),
                                                                                    platform.machine().lower())
    return f"{os_name}-{arch}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-tests", action="store_true")
    args = ap.parse_args()

    if not args.skip_tests:
        subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, check=True)

    collect = ["sherpa_onnx", "_soundfile_data"] + (["pyaudiowpatch"] if IS_WIN else ["_sounddevice_data"])
    cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--name", "TransVoice", "--console",
           "--distpath", str(DIST), "--workpath", str(ROOT / "build"), "--specpath", str(ROOT / "build")]
    for pkg in collect:
        try:
            __import__(pkg)
        except ImportError:  # e.g. _sounddevice_data is absent on Linux, where PortAudio comes from the system
            continue
        cmd += ["--collect-all", pkg]
    for mod in EXCLUDE:
        cmd += ["--exclude-module", mod]
    subprocess.run(cmd + [str(ROOT / "packaging" / "entry.py")], cwd=ROOT, check=True)

    app = DIST / "TransVoice"
    if sys.platform.startswith("linux"):
        drop_linux_system_libs(app)
    for f in ("README.md", "glossary.example.json"):
        shutil.copy(ROOT / f, app / f)
    if IS_WIN:
        (app / "TransVoice.bat").write_text(
            '@echo off\r\nrem Double-click to translate system audio into Vietnamese. Edit the arguments to change.\r\n'
            'chcp 65001 > nul\r\n"%~dp0TransVoice.exe" --me vi %*\r\n', encoding="ascii")
    else:
        launcher = app / "run.sh"
        launcher.write_text('#!/bin/sh\n# Translate system audio into Vietnamese; edit the arguments to change.\n'
                            'cd "$(dirname "$0")" && exec ./TransVoice --me vi "$@"\n', encoding="utf-8")
        launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    base = DIST / f"TransVoice-{platform_tag()}"
    # zip for Windows; tar.gz elsewhere because it keeps the executable bits
    archive = shutil.make_archive(str(base), "zip" if IS_WIN else "gztar", root_dir=DIST, base_dir="TransVoice")
    print(f"{Path(archive).stat().st_size / 2**20:.0f} MB  {archive}")


if __name__ == "__main__":
    main()
