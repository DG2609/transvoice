@echo off
rem TransVoice launcher - see README.md
chcp 65001 > nul
set PYTHONUTF8=1
"%~dp0.venv\Scripts\python.exe" -m transvoice %*
