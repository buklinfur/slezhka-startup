from collections import deque
from typing import Deque, List, Optional
from .datamodel import ModelOutput
import time


class StatsCollector:
    """
    Простой буфер (скользящее окно) для ModelOutput'ов.
    maxlen — количество последних записей (не кадров — записей лиц).
    """
    def __init__(self, maxlen: int = 300):
        self.window: Deque[ModelOutput] = deque(maxlen=maxlen)
        self.last_frame_time: Optional[float] = None
        self._frame_count = 0
        self._frame_start_ts: Optional[float] = None

    def add(self, items: List[ModelOutput]):
        """
        Добавляет список ModelOutput (обычно все лица текущего кадра).
        """
        now = time.time()
        if self._frame_start_ts is None:
            self._frame_start_ts = now
        self.last_frame_time = now
        self._frame_count += 1

        for it in items:
            self.window.append(it)

    def snapshot(self) -> List[ModelOutput]:
        return list(self.window)

    def clear(self):
        self.window.clear()
        self._frame_count = 0
        self._frame_start_ts = None

    def fps(self) -> Optional[float]:
        """
        Примитивная оценка FPS обработки: кадры / секунда с момента первого добавления.
        """
        if self._frame_start_ts is None:
            return None
        elapsed = time.time() - self._frame_start_ts
        if elapsed <= 0:
            return None
        return self._frame_count / elapsed