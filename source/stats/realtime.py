from typing import List, Any, Tuple
import time
import numpy as np
from .datamodel import ModelOutput
from .collector import StatsCollector
from .metrics import compute_metrics
from pathlib import Path
import torch


def _to_numpy(x):
    """Привести возможный torch.Tensor -> numpy, переводя на cpu."""
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    try:
        import torch as _t
        if isinstance(x, _t.Tensor):
            if x.device.type != "cpu":
                x = x.cpu()
            return x.detach().numpy()
    except Exception:
        pass
    return np.array(x)


def _normalize_bbox(bbox, image_shape):
    """
    Преобразует bbox в формат (x1,y1,x2,y2) в пикселях.
    Поддерживаем разные варианты входа:
    - normalized [x1, y1, x2, y2] (от 0 до 1)
    - absolute [x1, y1, x2, y2] (пиксели, int/float)
    - torch tensor, numpy array
    """
    if bbox is None:
        return None
    arr = _to_numpy(bbox)
    arr = np.asarray(arr).astype(float).flatten()
    h, w = image_shape[:2]

    # если значения в диапазоне [0,1] — считаем нормализованными
    if np.all(arr <= 1.0 + 1e-6):
        x1 = int(arr[0] * w)
        y1 = int(arr[1] * h)
        x2 = int(arr[2] * w)
        y2 = int(arr[3] * h)
    else:
        x1, y1, x2, y2 = arr[:4]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # sanitize
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    return (x1, y1, x2, y2)


def merge_outputs(image, bbox_list, gaze_tuple, emotion_list) -> List[ModelOutput]:
    """
    Собирает результаты в список ModelOutput.
    - bbox_list: список bbox (формат свободный)
    - gaze_tuple: (pitch_array, yaw_array) или None
    - emotion_list: list[dict] returned by DeepFace.analyze (or None)
    """
    outputs = []
    n = 0
    lens = [len(bbox_list) if bbox_list is not None else 0,
            len(gaze_tuple[0]) if (gaze_tuple is not None and gaze_tuple[0] is not None) else 0,
            len(emotion_list) if emotion_list is not None else 0]
    # We'll take minimal available count to pair things by index
    n = min([l for l in lens if l is not None] + [0])
    if n == 0:
        # try to handle if there's at least bbox_list
        if bbox_list:
            n = len(bbox_list)
        else:
            return outputs

    # Normalize arrays
    pitch_arr = None
    yaw_arr = None
    if gaze_tuple is not None:
        p = _to_numpy(gaze_tuple[0])
        y = _to_numpy(gaze_tuple[1])
        if p is not None and y is not None:
            pitch_arr = np.ravel(p)
            yaw_arr = np.ravel(y)

    for i in range(n):
        bbox = bbox_list[i] if bbox_list is not None and i < len(bbox_list) else None
        bbox_px = _normalize_bbox(bbox, image.shape) if bbox is not None else None

        # confidence: try common patterns
        conf = None
        # if bbox is object with score attribute / tensor
        try:
            if hasattr(bbox, "score"):
                conf = float(bbox.score)
        except Exception:
            pass

        # emotion
        emo_scores = None
        emo_label = None
        if emotion_list is not None and i < len(emotion_list):
            emo = emotion_list[i]
            # DeepFace analyze returns dict possibly with key 'emotion' and 'dominant_emotion'
            if isinstance(emo, dict):
                emo_scores = emo.get("emotion") or emo.get("scores") or None
                emo_label = emo.get("dominant_emotion") or emo.get("label") or None

        # gaze
        gaze_val = None
        if pitch_arr is not None and yaw_arr is not None and i < len(pitch_arr) and i < len(yaw_arr):
            gaze_val = (float(pitch_arr[i]), float(yaw_arr[i]))

        mo = ModelOutput(
            timestamp=time.time(),
            bbox=bbox_px,
            confidence=conf,
            gaze=gaze_val,
            emotion_scores=emo_scores,
            emotion_label=emo_label
        )
        outputs.append(mo)

    return outputs


def process_frame(frame, face_detector, gaze_detector, emo_classifier, preprocessor=None) -> Tuple[List[ModelOutput], dict]:
    """
    Обрабатывает один кадр:
    - вызывает face_detector.forward (ожидаем preprocessed tensor или raw?).
      Мы предполагаем интерфейс как у ваших модулей:
        face_detector.forward(img_tensor) -> bbox_list
        face_detector.make_face_batch(image, bbox_list, pre_process) -> face_batch (tensor)
        gaze_detector.forward(face_batch) -> pitch, yaw  (numpy arrays)
        emo_classifier.predict(face_batch) -> list[dict]
    Возвращает (list(ModelOutput), timing_info)
    """
    t0 = time.time()
    # 1) prepare image for face detector: your face_detector.forward expects tensor processed with pre_process
    if preprocessor is None:
        # try to use default preprocessor from gaze module if available
        try:
            from ..gaze_estimation import pre_process as default_pre
            preprocessor = default_pre
        except Exception:
            preprocessor = None

    # face detection: your implementation expects preprocessed tensor
    try:
        img_tensor = preprocessor(frame).to(next(gaze_detector.model.parameters()).device) if preprocessor is not None else None
    except Exception:
        # fallback: try to call forward with np array (some YOLO wrappers accept either)
        img_tensor = None

    # Run face detector
    if img_tensor is not None:
        bbox_list = face_detector.forward(img_tensor)
    else:
        # try passing an np array / cv2 image if detector supports it
        try:
            bbox_list = face_detector.forward(frame)
        except Exception:
            bbox_list = []

    t1 = time.time()

    # Make face batch for gaze/emotion
    face_batch = []
    try:
        face_batch = face_detector.make_face_batch(frame, bbox_list, preprocessor)
        # ensure tensor on device
        if isinstance(face_batch, (list, tuple)):
            # some impl might return list of tensors
            face_batch = torch.stack(face_batch) if len(face_batch) else torch.tensor([])
    except Exception:
        face_batch = torch.tensor([])

    t2 = time.time()

    # Gaze
    gaze_pitch, gaze_yaw = None, None
    try:
        if isinstance(face_batch, (list, tuple)) or (hasattr(face_batch, "shape") and getattr(face_batch, "shape")[0] > 0):
            # send to device expected by gaze_detector
            try:
                dev = next(gaze_detector.model.parameters()).device
                face_batch_device = face_batch.to(dev)
            except Exception:
                face_batch_device = face_batch
            gaze_pitch, gaze_yaw = gaze_detector.forward(face_batch_device)
    except Exception:
        gaze_pitch, gaze_yaw = None, None

    t3 = time.time()

    # Emotions
    emotion_list = None
    try:
        # emo_classifier.predict expects numpy array or tensor or list
        emotion_list = emo_classifier.predict(face_batch)
    except Exception:
        # sometimes DeepFace or classifier may want lists of np imgs
        emotion_list = []
        try:
            # attempt conversion
            fb = _to_numpy(face_batch)
            if fb is not None:
                for i in range(getattr(fb, 0, fb.shape[0] if hasattr(fb, 'shape') else 0)):
                    emotion_list.append({})
        except Exception:
            emotion_list = []

    t4 = time.time()

    outputs = merge_outputs(frame, bbox_list or [], (gaze_pitch, gaze_yaw), emotion_list or [])
    timings = {
        "t_detect": t1 - t0,
        "t_crop": t2 - t1,
        "t_gaze": t3 - t2,
        "t_emo": t4 - t3,
        "t_total": t4 - t0
    }
    return outputs, timings
