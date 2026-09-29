"""PyInstaller entry point for TransVoice.exe."""
import sys
import traceback

from transvoice.__main__ import main, own_console

if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 - a double-clicked console window would close before it can be read
        traceback.print_exc()
        if own_console():
            input("Nhấn Enter để đóng.")
        sys.exit(1)
