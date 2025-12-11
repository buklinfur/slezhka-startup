from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple
import time


@dataclass
class ModelOutput:
    """
    Унифицированный контейнер для одного детектированного лица / объекта на кадре.
    Поля могут быть None если модель не дала результат.
    """
    timestamp: float = field(default_factory=time.time)
    bbox: Optional[Tuple[int, int, int, int]] = None  # (x1,y1,x2,y2) в пикселях
    confidence: Optional[float] = None                # confidence детектора лица
    gaze: Optional[Tuple[float, float]] = None        # (pitch_rad, yaw_rad) - радианы
    emotion_scores: Optional[Dict[str, float]] = None # вероятности по эмоциям
    emotion_label: Optional[str] = None               # доминирующая эмоция
    meta: Optional[Dict] = None 