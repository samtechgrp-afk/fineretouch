"""Peak resident memory of this process while a block runs (no extra dependencies).

A sampling thread reads the current resident set every few milliseconds: /proc on Linux, the
Win32 working set on Windows, and getrusage's lifetime peak elsewhere (macOS; an upper bound).
Numbers are the process total (Python, NumPy, OpenCV and MediaPipe buffers), not only the engine.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from contextlib import contextmanager


def _rss_linux() -> int | None:
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except Exception:
        return None


def _rss_windows() -> int | None:
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        h = ctypes.windll.kernel32.GetCurrentProcess()
        if ctypes.windll.psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb):
            return int(pmc.WorkingSetSize)
    except Exception:
        return None
    return None


def _rss_rusage_peak() -> int | None:
    try:
        import resource

        v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(v if sys.platform == "darwin" else v * 1024)
    except Exception:
        return None


def current_rss() -> int | None:
    if sys.platform.startswith("linux"):
        return _rss_linux()
    if sys.platform == "win32":
        return _rss_windows()
    return _rss_rusage_peak()


class Probe:
    peak_bytes: int | None = None
    start_bytes: int | None = None
    method: str = "sampled"

    @property
    def peak_mb(self) -> float | None:
        return None if self.peak_bytes is None else round(self.peak_bytes / 2**20, 1)

    @property
    def added_mb(self) -> float | None:
        if self.peak_bytes is None or self.start_bytes is None:
            return None
        return round(max(0, self.peak_bytes - self.start_bytes) / 2**20, 1)


@contextmanager
def peak_memory(interval: float = 0.01):
    """with peak_memory() as p: ...  ->  p.peak_mb (process peak), p.added_mb (above the start)."""
    p = Probe()
    p.start_bytes = current_rss()
    if p.start_bytes is None:
        p.method = "unavailable"
        yield p
        return
    if not (sys.platform.startswith("linux") or sys.platform == "win32"):
        p.method = "lifetime-peak"
    peak = [p.start_bytes]
    stop = threading.Event()

    def run():
        while not stop.is_set():
            v = current_rss()
            if v is not None and v > peak[0]:
                peak[0] = v
            time.sleep(interval)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    try:
        yield p
    finally:
        stop.set()
        t.join()
        v = current_rss()
        p.peak_bytes = max(peak[0], v or 0)
