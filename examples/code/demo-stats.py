# prep part (без изменений)
import sys
from pathlib import Path
import importlib.util
import pprint

cwd = Path.cwd().resolve()
root = cwd
while not (root / "source").exists():
    if root.parent == root:
        raise RuntimeError("Не найден каталог 'source' в родителях текущей директории. Запустите из корня репо.")
    root = root.parent

SRC = root / "source"
print("Project root:", root)
print("Source path:", SRC)
sys.path.insert(0, str(SRC))

def alias_package(name: str, path: Path):
    init = path / "__init__.py"
    if not path.exists():
        raise RuntimeError(f"Path for alias {name} does not exist: {path}")
    if init.exists():
        spec = importlib.util.spec_from_file_location(name, str(init))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[name] = module
    else:
        import types
        m = types.ModuleType(name)
        sys.modules[name] = m

alias_package("utils", SRC / "utils")
alias_package("models", SRC / "models")
print("Aliased packages: utils, models ->", SRC / "utils", SRC / "models")


# code part
import cv2
import torch
from source.gaze_estimation import yolo_face, mobile_gaze, pre_process
from source.emo_classifier import EmoClassifier
from source.stats.realtime import process_frame
from source.stats.collector import StatsCollector
from source.stats.metrics import compute_metrics

IMG = root / "examples" / "images" / "image.png"
if not IMG.exists():
    raise RuntimeError(f"Image not found: {IMG}")

print("Image:", IMG)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

face_detector = yolo_face(device)
gaze_detector = mobile_gaze(device)
emo_clf = EmoClassifier(detector_backend='mtcnn', enforce_detection=False)

bgr = cv2.imread(str(IMG))
if bgr is None:
    raise RuntimeError("cv2.imread вернул None — проверь путь или формат картинки")

outputs, timings = process_frame(bgr, face_detector, gaze_detector, emo_clf, preprocessor=pre_process)

print("\n=== Raw outputs (ModelOutput list) ===")
pprint.pprint(outputs)

print("\n=== Timings ===")
pprint.pprint(timings)

metrics = compute_metrics(outputs)
print("\n=== Aggregated metrics ===")
pprint.pprint(metrics)


# visualization 

bbox_list = []
pitch_list = []
yaw_list = []
emotions_list = []

for mo in outputs:
    if not mo.bbox:
        continue

    bbox_list.append(mo.bbox)

    if hasattr(mo, "gaze") and mo.gaze is not None:
        g = mo.gaze
        pitch_list.append(float(g[0]))
        yaw_list.append(float(g[1]))
    elif hasattr(mo, "pitch") and hasattr(mo, "yaw"):
        pitch_list.append(float(mo.pitch))
        yaw_list.append(float(mo.yaw))
    else:
        pitch_list.append(0.0)
        yaw_list.append(0.0)

    dominant = getattr(mo, "emotion_label", "") or ""
    scores = getattr(mo, "emotion_scores", None)

    if dominant or scores:
        if scores is None:
            emotions_list.append({"dominant_emotion": dominant, "emotion": {dominant: 1.0}})
        else:
            emotions_list.append({
                "dominant_emotion": dominant,
                "emotion": {k: float(v) for k, v in scores.items()}
            })
    else:
        emotions_list.append({"dominant_emotion": "", "emotion": {}})


try:
    gaze_detector.draw_result(bgr, pitch_list, yaw_list, bbox_list)
except Exception as e:
    print("Warning: gaze draw failed:", e)

final_image = emo_clf.draw_result(bgr, bbox_list, emotions_list)

out_path = root / "examples" / "images" / "image_with_stats.png"
cv2.imwrite(str(out_path), final_image)
print("\nWrote visualization to:", out_path)
