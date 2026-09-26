"""The few things that differ between Windows, macOS and Linux."""
import atexit
import os
import platform
import signal
import subprocess
import sys

IS_WIN = sys.platform == "win32"
IS_MAC = sys.platform == "darwin"
IS_LINUX = sys.platform.startswith("linux")


def lower_own_priority() -> None:
    """Never slow down the call or game being translated."""
    import psutil

    p = psutil.Process()
    p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if IS_WIN else 10)


def child_process_kwargs(low_priority: bool) -> dict:
    """Popen arguments for helper processes (llama-server): no console window, lower priority,
    and on Linux die together with the parent."""
    if IS_WIN:
        flags = subprocess.CREATE_NO_WINDOW | (subprocess.BELOW_NORMAL_PRIORITY_CLASS if low_priority else 0)
        return {"creationflags": flags}

    def preexec() -> None:
        if low_priority:
            os.nice(10)
        if IS_LINUX:
            import ctypes

            ctypes.CDLL("libc.so.6").prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG

    return {"preexec_fn": preexec}


_JOBS: list = []


def kill_with_parent(proc: subprocess.Popen) -> None:
    """Make sure an abandoned llama-server never keeps ~2 GB of RAM after the app exits or crashes."""
    if IS_WIN:
        _windows_job(proc.pid)
    else:
        # Linux also has PR_SET_PDEATHSIG (above) for crashes; macOS only gets a clean-exit hook.
        atexit.register(lambda: proc.poll() is None and proc.terminate())


def _windows_job(pid: int) -> None:
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.OpenProcess.restype = wintypes.HANDLE

    class LIMIT(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class IO(ctypes.Structure):
        _fields_ = [(n, ctypes.c_uint64) for n in ("r", "w", "o", "rb", "wb", "ob")]

    class EXTENDED(ctypes.Structure):
        _fields_ = [("Basic", LIMIT), ("Io", IO), ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    job = k32.CreateJobObjectW(None, None)
    info = EXTENDED()
    info.Basic.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info))  # ExtendedLimitInformation
    proc = k32.OpenProcess(0x0001 | 0x0100, False, pid)  # PROCESS_TERMINATE | PROCESS_SET_QUOTA
    k32.AssignProcessToJobObject(job, proc)
    k32.CloseHandle(proc)
    _JOBS.append(job)  # the handle must stay open for the life of this process


def llama_asset() -> str:
    """llama.cpp CPU build for this OS/CPU (release b11179)."""
    arch = platform.machine().lower()
    if IS_WIN:
        return "llama-b11179-bin-win-cpu-x64.zip"
    if IS_MAC:
        return "llama-b11179-bin-macos-arm64.tar.gz" if arch in ("arm64", "aarch64") else "llama-b11179-bin-macos-x64.tar.gz"
    return "llama-b11179-bin-ubuntu-arm64.tar.gz" if arch in ("arm64", "aarch64") else "llama-b11179-bin-ubuntu-x64.tar.gz"
