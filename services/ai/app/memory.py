"""Memory admission for full-resolution renders (v0.12).

A full-resolution render holds several float32 copies of the photo (source, blur bands, working
copy, masks). Measured on a 20 MP portrait: about 90 MB per megapixel with light and colour
edits and about 135 MB per megapixel with the skin tools on (pipeline strips included). Three
24 MP renders at once would need about 10 GB, which on a 16 GB laptop pushes the editor and the
browser into swap. The gate admits a render only while the estimated memory of the renders in
flight stays inside a budget; a render that does not fit waits for one to finish. One render is
always admitted, however large, so nothing can wait forever.

Budget: FINERETOUCH_RENDER_MEMORY_MB if set, else 60% of physical memory, else (memory unknown)
no limit beyond the callers' own concurrency.
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass

#: Peak bytes per output pixel of a render (skin tools on; measured, plus decode and encode).
BYTES_PER_PIXEL = 150
#: Fixed overhead per render (models' scratch, encoder buffers).
BASE_BYTES = 200 * 2**20


def total_memory() -> int | None:
    """Physical memory in bytes, or None when it cannot be read."""
    try:
        import psutil  # type: ignore[import-not-found]

        return int(psutil.virtual_memory().total)
    except Exception:
        pass
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/meminfo", encoding="ascii") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) * 1024
        except OSError:
            return None
    if sys.platform == "win32":

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        st = MemoryStatus()
        st.dwLength = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):  # type: ignore[attr-defined]
            return int(st.ullTotalPhys)
        return None
    if sys.platform == "darwin":
        try:
            return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
        except (ValueError, OSError, AttributeError):
            return None
    return None


def render_estimate(width: int, height: int) -> int:
    """Estimated peak bytes of rendering a width x height photo."""
    return BASE_BYTES + BYTES_PER_PIXEL * max(1, width) * max(1, height)


def default_budget() -> int | None:
    env = os.environ.get("FINERETOUCH_RENDER_MEMORY_MB")
    if env:
        try:
            return max(256, int(env)) * 2**20
        except ValueError:
            pass
    total = total_memory()
    return int(total * 0.6) if total else None


@dataclass
class GateStats:
    in_flight: int
    reserved: int
    budget: int | None
    peak_in_flight: int
    waited: int


class MemoryGate:
    """Counts the estimated bytes of renders in flight against a budget (thread-safe)."""

    def __init__(self, budget: int | None) -> None:
        self.budget = budget
        self._cv = threading.Condition()
        self._reserved = 0
        self._in_flight = 0
        self._peak = 0
        self._waited = 0

    @contextmanager
    def admit(self, estimate: int):
        with self._cv:
            if not self._fits(estimate):
                self._waited += 1
                self._cv.wait_for(lambda: self._fits(estimate))
            self._reserved += estimate
            self._in_flight += 1
            self._peak = max(self._peak, self._in_flight)
        try:
            yield
        finally:
            with self._cv:
                self._reserved -= estimate
                self._in_flight -= 1
                self._cv.notify_all()

    def _fits(self, estimate: int) -> bool:
        return self.budget is None or self._in_flight == 0 or self._reserved + estimate <= self.budget

    def stats(self) -> GateStats:
        with self._cv:
            return GateStats(self._in_flight, self._reserved, self.budget, self._peak, self._waited)


render_gate = MemoryGate(default_budget())
