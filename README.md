# Emo Radar reklama edition

Python package for **real-time emotion, gaze, and face tracking**.
Designed as a startup-grade prototype to provide modular components for video-based human behavior analysis.

The package offers reusable core functions for face detection, gaze estimation, emotion classification, and convenient example pipelines for video and webcam inputs.

---

## Project Structure

```
Emo-Radar/
├── examples/
│   ├── images/
│   │   ├── checkmeout.png
│   │   ├── checkmeout_with_emotions.png
│   │   └── image.png
│   └── notebooks/
├── source/
│   ├── emo_classifier.py
│   ├── gaze_estimation.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── mobilenet.py
│   │   ├── mobileone.py
│   │   └── resnet.py
│   ├── utils/
│   │   ├── datasets.py
│   │   ├── helpers.py
│   │   └── __init__.py
│   └── weights/
│       ├── mobilenetv2.pt
│       ├── mobileone_s0.pt
│       ├── resnet18.pt
│       └── yolov8face.pt
├── pyproject.toml
├── README.md
└── README-examples/
```

---

## Modules Overview

### `emo_classifier.py`

Provides real-time emotion detection based on face crops.

* `EmoClassifier(detector_backend='mtcnn', enforce_detection=False)` – initialize classifier.
* `predict(face_batch)` – predict emotions for a batch of faces.
* `draw_result(frame, bbox_list, emotions)` – overlay emotion labels on video frames.

### `gaze_estimation.py`

Provides gaze estimation using MobileNet or MobileOne-based models.

* `yolo_face(device)` – detect faces using YOLO.
* `mobile_gaze(device)` – estimate gaze pitch and yaw.
* `draw_result(frame, pitch, yaw, bbox_list)` – overlay gaze directions on frames.

### `models/`

Contains neural network definitions for gaze estimation:

* `mobilenet.py`, `mobileone.py`, `resnet.py` – model architectures.

### `utils/`

General helper functions:

* `datasets.py` – dataset loading and preprocessing.
* `helpers.py` – miscellaneous utilities (image processing, normalization, batching).

### `weights/`

Pretrained model weights for gaze and face detection.

---

## Installation

```bash
git clone <repo_url>
cd Emo-Radar
python -m venv .venv
source .venv/bin/activate  # Linux / macOS
# or
.venv\Scripts\activate    # Windows

pip install .
```

Install extra dependencies for development:

```bash
pip install .[dev]
```

---

## Usage Examples

### 1. Real-time Webcam Tracking

```python
import cv2
from source.emo_classifier import EmoClassifier
from source.gaze_estimation import yolo_face, mobile_gaze

device = 'cuda'
face_detector = yolo_face(device)
gaze_detector = mobile_gaze(device)
emo_classifier = EmoClassifier(detector_backend='mtcnn', enforce_detection=False)

cap = cv2.VideoCapture(0)  # webcam
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Face detection
    bbox_list = face_detector.forward(frame)
    face_batch = face_detector.make_face_batch(frame, bbox_list)

    if face_batch.shape[0]:
        # Gaze estimation
        pitch, yaw = gaze_detector.forward(face_batch.to(device))
        gaze_detector.draw_result(frame, pitch, yaw, bbox_list)

        # Emotion recognition
        emotions = emo_classifier.predict(face_batch)
        frame = emo_classifier.draw_result(frame, bbox_list, emotions)

    cv2.imshow('Emo Radar', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### 2. Video Input Processing

```python
cap = cv2.VideoCapture('input_video.webm')
out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 30.0, (640, 480))

# Pipeline same as webcam example
```

---

## Project Vision

Emo Radar is designed as a **foundational startup prototype** for real-time behavioral analytics.
It aims to grow into a modular and robust platform for video-based emotion and gaze tracking, enabling applications in:

* Human-computer interaction
* Market research and UX studies
* Video analytics for media and education

The architecture is intentionally modular: face detection, gaze estimation, and emotion recognition can be extended independently.

---

## Testing

```bash
pytest -v
```

Covers:

* Face and gaze detection pipeline
* Emotion classification
* Video and image processing utilities
