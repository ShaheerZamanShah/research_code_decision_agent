"""Timing and performance utilities."""

import time
import asyncio
from contextlib import asynccontextmanager, contextmanager
from typing import Dict, List
from dataclasses import dataclass, field


@dataclass
class TimingRecord:
    name: str
    start: float
    end: float = 0.0

    @property
    def elapsed(self) -> float:
        return self.end - self.start


class PerformanceTracker:
    """Thread-safe performance tracker for agent steps."""

    def __init__(self):
        self.records: List[TimingRecord] = []
        self._active: Dict[str, float] = {}

    def start(self, name: str):
        self._active[name] = time.perf_counter()

    def stop(self, name: str) -> float:
        if name in self._active:
            elapsed = time.perf_counter() - self._active.pop(name)
            self.records.append(TimingRecord(name=name, start=0, end=elapsed))
            return elapsed
        return 0.0

    @contextmanager
    def track(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.records.append(TimingRecord(name=name, start=start, end=start + elapsed))

    @asynccontextmanager
    async def atrack(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.records.append(TimingRecord(name=name, start=start, end=start + elapsed))

    def summary(self) -> Dict[str, float]:
        return {r.name: r.elapsed for r in self.records}

    def total(self) -> float:
        return sum(r.elapsed for r in self.records)
