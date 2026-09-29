import os
import sys

from transvoice import paths


def test_source_checkout_uses_the_repo(monkeypatch):
    monkeypatch.delenv("TRANSVOICE_HOME", raising=False)
    monkeypatch.delattr(sys, "frozen", raising=False)
    assert (paths._root() / "transvoice" / "paths.py").exists()


def test_portable_app_keeps_models_next_to_the_executable(monkeypatch, tmp_path):
    monkeypatch.delenv("TRANSVOICE_HOME", raising=False)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "TransVoice.exe"))
    assert paths._root() == tmp_path


def test_read_only_install_uses_the_user_data_folder(monkeypatch, tmp_path):
    # /opt/transvoice from the .deb, or the app inside a mounted .dmg: nothing can be written next to it.
    monkeypatch.delenv("TRANSVOICE_HOME", raising=False)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "opt" / "TransVoice"))
    monkeypatch.setattr(paths, "_writable", lambda folder: False)
    assert paths._root() == paths._user_data_dir()


def test_models_already_next_to_the_executable_win(monkeypatch, tmp_path):
    monkeypatch.delenv("TRANSVOICE_HOME", raising=False)
    (tmp_path / "models").mkdir()
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "TransVoice.exe"))
    monkeypatch.setattr(paths, "_writable", lambda folder: False)
    assert paths._root() == tmp_path


def test_environment_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("TRANSVOICE_HOME", str(tmp_path))
    assert paths._root() == tmp_path
    assert os.environ["TRANSVOICE_HOME"] == str(tmp_path)
