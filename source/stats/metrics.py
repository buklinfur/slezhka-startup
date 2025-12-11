from typing import List, Dict, Any
import numpy as np
from .datamodel import ModelOutput
import math


def _safe_list(values):
    return [v for v in values if v is not None]


def mean_confidence(window: List[ModelOutput]) -> float:
    confs = _safe_list([m.confidence for m in window])
    return float(np.mean(confs)) if confs else float('nan')


def emotion_aggregates(window: List[ModelOutput], positive_keys=('happy',), negative_keys=('angry', 'disgust', 'sad', 'fear')) -> Dict[str, float]:
    """
    Возвращает простую агрегацию эмоций:
      - средние score по каждой эмоции (всех, что встретились)
      - ratio_positive = sum(pos_scores) / sum(all)
      - ratio_negative = sum(neg_scores) / sum(all)
      - avg_engagement = mean(max_score - min_score)  (простая мера "эмоционального всплеска")
    """
    all_scores = []
    pos_sum = 0.0
    neg_sum = 0.0
    total_sum = 0.0
    engagement_vals = []

    for m in window:
        if not m.emotion_scores:
            continue
        scores = m.emotion_scores
        vals = list(scores.values())
        total_sum += sum(vals)
        pos_sum += sum(scores.get(k, 0.0) for k in positive_keys)
        neg_sum += sum(scores.get(k, 0.0) for k in negative_keys)
        engagement_vals.append((max(vals) - min(vals)))
        all_scores.append(scores)

    # compute per-emotion mean
    mean_by_em = {}
    if all_scores:
        keys = set().union(*(d.keys() for d in all_scores))
        for k in keys:
            mean_by_em[k] = float(np.mean([d.get(k, 0.0) for d in all_scores]))

    ratio_positive = (pos_sum / total_sum) if total_sum > 0 else float('nan')
    ratio_negative = (neg_sum / total_sum) if total_sum > 0 else float('nan')
    avg_engagement = float(np.mean(engagement_vals)) if engagement_vals else float('nan')

    return {
        "mean_by_emotion": mean_by_em,
        "ratio_positive": ratio_positive,
        "ratio_negative": ratio_negative,
        "avg_engagement": avg_engagement
    }


def attention_stats(window: List[ModelOutput], yaw_thresh_deg: float = 15.0, pitch_thresh_deg: float = 20.0) -> Dict[str, Any]:
    """
    Оценивает, какую долю времени люди "смотрят прямо" (тут - упрощение).
    Т.е. если |yaw| < yaw_thresh_deg and |pitch| < pitch_thresh_deg считаем внимание=1.
    Ожидаем gaze в радианах.
    """
    total = 0
    attentive = 0
    yaw_vals = []
    pitch_vals = []
    for m in window:
        if m.gaze is None:
            continue
        pitch, yaw = m.gaze
        pitch_deg = abs(math.degrees(pitch))
        yaw_deg = abs(math.degrees(yaw))
        total += 1
        yaw_vals.append(yaw_deg)
        pitch_vals.append(pitch_deg)
        if yaw_deg <= yaw_thresh_deg and pitch_deg <= pitch_thresh_deg:
            attentive += 1

    return {
        "attention_ratio": (attentive / total) if total > 0 else float('nan'),
        "count_with_gaze": total,
        "mean_yaw_deg": float(np.mean(yaw_vals)) if yaw_vals else float('nan'),
        "mean_pitch_deg": float(np.mean(pitch_vals)) if pitch_vals else float('nan'),
    }


def compute_metrics(window: List[ModelOutput]) -> Dict[str, Any]:
    """
    Собирает основные метрики в единый словарь.
    """
    metrics = {}
    metrics["mean_confidence"] = mean_confidence(window)
    metrics.update(emotion_aggregates(window))
    metrics.update(attention_stats(window))
    metrics["total_samples"] = len(window)
    return metrics