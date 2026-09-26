"""Samples CPU time and peak RAM of a set of processes while a benchmark runs."""
import threading
import time

import psutil


def limit_cpus(spec: str | None) -> None:
    """Pin this process (and the children it starts later, e.g. llama-server) to the given logical CPUs.

    On the i5-14600K, logical CPUs 12-19 are the E-cores; four of them approximate an office PC/laptop.
    """
    if spec:
        psutil.Process().cpu_affinity([int(c) for c in spec.split(",")])


class ResourceMonitor:
    """Context manager: tracks peak RSS (MB) and average CPU cores used by `pids`."""

    def __init__(self, pids: list[int], interval: float = 0.2):
        self.procs = [psutil.Process(p) for p in pids]
        self.interval = interval
        self.peak_rss_mb = 0.0
        self._stop = threading.Event()

    def _cpu_seconds(self) -> float:
        total = 0.0
        for p in self.procs:
            try:
                t = p.cpu_times()
                total += t.user + t.system
            except psutil.NoSuchProcess:
                pass
        return total

    def _sample(self) -> None:
        while not self._stop.is_set():
            rss = 0
            for p in self.procs:
                try:
                    rss += p.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            self.peak_rss_mb = max(self.peak_rss_mb, rss / 2**20)
            self._stop.wait(self.interval)

    def __enter__(self):
        self._cpu0 = self._cpu_seconds()
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self.wall_s = time.perf_counter() - self._t0
        self.cpu_s = self._cpu_seconds() - self._cpu0
        self._stop.set()
        self._thread.join()

    @property
    def avg_cores(self) -> float:
        return self.cpu_s / self.wall_s if self.wall_s else 0.0
