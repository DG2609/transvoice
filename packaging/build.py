"""Build TransVoice for the current OS with PyInstaller.

  python packaging/build.py            # runs the tests, then builds
Output in dist/:
  Windows  TransVoice-windows-x64.zip (folder) and TransVoice-windows-x64.exe (one file, double-click to run)
  Linux    TransVoice-linux-x64.tar.gz and transvoice_<version>_amd64.deb (installs /opt/transvoice, `transvoice`)
  macOS    TransVoice-macos-arm64.tar.gz and TransVoice-macos-arm64.dmg
Models are not bundled: the first run downloads them (~2.9 GB) next to the executable, or into the user's data
folder when the app is installed read-only (.deb, .dmg).
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


def version() -> str:
    text = (ROOT / "transvoice" / "__init__.py").read_text(encoding="utf-8")
    return text.split('__version__ = "', 1)[1].split('"', 1)[0]


def pyinstaller(onefile: bool) -> None:
    collect = ["sherpa_onnx", "_soundfile_data"] + (["pyaudiowpatch"] if IS_WIN else ["_sounddevice_data"])
    work = ROOT / "build" / ("onefile" if onefile else "onedir")
    cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--name", "TransVoice", "--console",
           "--distpath", str(DIST), "--workpath", str(work), "--specpath", str(work)]
    if onefile:
        cmd.append("--onefile")
    for pkg in collect:
        try:
            __import__(pkg)
        except ImportError:  # e.g. _sounddevice_data is absent on Linux, where PortAudio comes from the system
            continue
        cmd += ["--collect-all", pkg]
    for mod in EXCLUDE:
        cmd += ["--exclude-module", mod]
    subprocess.run(cmd + [str(ROOT / "packaging" / "entry.py")], cwd=ROOT, check=True)


def build_deb(app: Path) -> Path:
    """Debian package: the app in /opt/transvoice, a `transvoice` command and a menu entry."""
    stage = ROOT / "build" / "deb"
    shutil.rmtree(stage, ignore_errors=True)
    shutil.copytree(app, stage / "opt" / "transvoice", symlinks=True)
    (stage / "usr" / "bin").mkdir(parents=True)
    cmd = stage / "usr" / "bin" / "transvoice"
    cmd.write_text('#!/bin/sh\n# Models and history go to ~/.local/share/transvoice (/opt is read-only).\n'
                   'exec /opt/transvoice/TransVoice "$@"\n', encoding="utf-8")
    cmd.chmod(0o755)
    (stage / "usr" / "share" / "applications").mkdir(parents=True)
    (stage / "usr" / "share" / "applications" / "transvoice.desktop").write_text(
        "[Desktop Entry]\nType=Application\nName=TransVoice\nComment=Live speech translation JA/EN/VI\n"
        "Exec=transvoice --me vi\nTerminal=true\nCategories=AudioVideo;Utility;\n", encoding="utf-8")
    (stage / "DEBIAN").mkdir()
    (stage / "DEBIAN" / "control").write_text(
        f"Package: transvoice\nVersion: {version()}\nArchitecture: amd64\n"
        "Maintainer: DG2609 <DG2609@users.noreply.github.com>\n"
        "Depends: libc6 (>= 2.27), libgomp1, libportaudio2, libx11-6\n"
        "Recommends: pulseaudio-utils\nSection: sound\nPriority: optional\n"
        "Description: Live speech translation between Japanese, English and Vietnamese\n"
        " Local and CPU-only: speech recognition (sherpa-onnx) and translation (HY-MT via llama.cpp).\n"
        " The first run downloads the models (~2.9 GB) into ~/.local/share/transvoice.\n", encoding="utf-8")
    deb = DIST / f"transvoice_{version()}_amd64.deb"
    subprocess.run(["dpkg-deb", "--root-owner-group", "--build", str(stage), str(deb)], check=True)
    return deb


def build_dmg(app: Path) -> Path:
    """Disk image with the app folder. The app is not signed: first open it with right-click > Open."""
    dmg = DIST / f"TransVoice-{platform_tag()}.dmg"
    dmg.unlink(missing_ok=True)
    subprocess.run(["hdiutil", "create", "-volname", "TransVoice", "-srcfolder", str(app), "-ov",
                    "-format", "UDZO", str(dmg)], check=True)
    return dmg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-tests", action="store_true")
    args = ap.parse_args()

    if not args.skip_tests:
        subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, check=True)

    pyinstaller(onefile=False)

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
    outputs = [Path(archive)]
    if IS_WIN:  # one file: download, double-click, done
        pyinstaller(onefile=True)
        exe = base.with_suffix(".exe")
        exe.unlink(missing_ok=True)
        (DIST / "TransVoice.exe").rename(exe)
        outputs.append(exe)
    elif IS_MAC:
        outputs.append(build_dmg(app))
    elif shutil.which("dpkg-deb"):
        outputs.append(build_deb(app))
    for out in outputs:
        print(f"{out.stat().st_size / 2**20:.0f} MB  {out}")


if __name__ == "__main__":
    main()
