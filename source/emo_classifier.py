from typing import Union, Dict, List

import cv2
import numpy as np
import torch
from deepface import DeepFace

from .gaze_estimation import yolo_face, mobile_gaze, pre_process


class EmoClassifier:
    EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

    def __init__(self,
                 detector_backend: str = 'opencv',
                 enforce_detection: bool = False,
                 align: bool = True):
        """
        Инициализация классификатора эмоций.

        Args:
            detector_backend: Бэкенд для детекции лиц ('opencv', 'retinaface', 'mtcnn', и т.д.)
            enforce_detection: Если True, выбрасывает исключение при отсутствии лица
            align: Выравнивать ли лицо перед анализом
        """
        self.detector_backend = detector_backend
        self.enforce_detection = enforce_detection
        self.align = align

    def process_image(self, img):
        img_processed = img.copy()

        if img_processed.ndim == 3 and img_processed.shape[0] in [1, 3, 4]:
            img_processed = np.transpose(img_processed, (1, 2, 0))

        if img_processed.shape[0] < 10 or img_processed.shape[1] < 10:
            return None

        if img_processed.dtype != np.uint8:
            if img_processed.max() <= 1.0:
                img_processed = (img_processed * 255).astype(np.uint8)
            elif img_processed.max() > 255:
                img_processed = np.clip(img_processed, 0, 255).astype(np.uint8)
            else:
                img_processed = img_processed.astype(np.uint8)

        if img_processed.ndim == 3 and img_processed.shape[2] == 3:
            img_processed = cv2.cvtColor(img_processed, cv2.COLOR_RGB2BGR)

        return img_processed

    def predict(self, input_data: Union[np.ndarray, torch.Tensor, List[np.ndarray]]) -> List[Dict]:
        if isinstance(input_data, torch.Tensor):
            if input_data.dim() == 4:
                images = [img for img in input_data.numpy(force=True)]
            elif input_data.dim() == 3:
                images = [input_data.numpy(force=True)]
            else:
                raise ValueError(f"Не поддерживаемая размерность тензора: {input_data.dim()}")

        elif isinstance(input_data, np.ndarray):
            if input_data.ndim == 4:
                images = [img for img in input_data]
            elif input_data.ndim == 3:
                images = [input_data]
            else:
                raise ValueError(f"Не поддерживаемая размерность массива: {input_data.ndim}")

        elif isinstance(input_data, list):
            if all(isinstance(img, np.ndarray) for img in input_data):
                images = input_data
            else:
                raise ValueError("Все элементы списка должны быть numpy массивами")
        else:
            raise TypeError(f"Не поддерживаемый тип входных данных: {type(input_data)}")

        results = []
        for img in images:
            processed_img = self.process_image(img)
            if processed_img is None:
                print("Warning: Too small crop, skipped it")
                continue
            result = DeepFace.analyze(
                img_path=processed_img,
                actions=['emotion'],
                detector_backend=self.detector_backend,
                enforce_detection=self.enforce_detection,
                align=self.align,
                silent=True
            )
            results.append(result[0])

        return results

    def draw_result(self, image, bbox_list, emotions: List[Dict]):
        """
        Отображает название эмоции и confidence в зоне bounding box на изображении.

        Parameters:
        -----------
        image : numpy.ndarray
            Входное изображение в формате BGR (как в OpenCV)
        bbox : tuple or list
            Bounding box в формате (x, y, w, h) или [x, y, w, h]
        emotion : str
            Название распознанной эмоции
        confidence : float
            Уровень уверенности распознавания (от 0.0 до 1.0)

        Returns:
        --------
        numpy.ndarray
            Изображение с отрисованным текстом
        """
        output_image = image.copy()
        for bbox, emotion in zip(bbox_list, emotions):
            x_min, y_min, x_max, y_max = map(int, bbox[:4])
            h = y_max - y_min

            dominant_emotion = emotion["dominant_emotion"]
            confidence_percent = emotion["emotion"][dominant_emotion]
            text = f"{dominant_emotion}: {confidence_percent:.1f}%"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            text_color = (0, 255, 0)
            bg_color = (0, 0, 0)

            (text_width, text_height), baseline = cv2.getTextSize(
                text, font, font_scale, font_thickness
            )

            padding = 5
            text_x = x_min
            text_y = y_min + h + text_height + padding

            if text_y + text_height > image.shape[0]:
                text_y = y_min - padding

            output_image = cv2.rectangle(
                output_image,
                (text_x, text_y - text_height - baseline),
                (text_x + text_width, text_y + baseline),
                bg_color,
                -1
            )

            output_image = cv2.putText(
                output_image,
                text,
                (text_x, text_y),
                font,
                font_scale,
                text_color,
                font_thickness,
                cv2.LINE_AA
            )

        return output_image

    def get_supported_emotions(self) -> List[str]:
        """
        Возвращает список поддерживаемых эмоций.
        """
        return self.EMOTIONS.copy()


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('using ' + device)

    unprocessed = cv2.imread('image.png')
    img = pre_process(unprocessed).to(device)

    face_detector = yolo_face(device)
    gaze_detector = mobile_gaze(device)

    bbox_list = face_detector.forward(img)
    face_batch = face_detector.make_face_batch(unprocessed, bbox_list, pre_process)
    pitch, yaw = gaze_detector.forward(face_batch.to(device))
    gaze_detector.draw_result(unprocessed, pitch, yaw, bbox_list)

    emo_classifier = EmoClassifier(
        detector_backend='mtcnn',
        enforce_detection=False
    )
    emotions = emo_classifier.predict(face_batch)
    final_image = emo_classifier.draw_result(unprocessed, bbox_list, emotions)

    cv2.imwrite('checkmeout_with_emotions.png', final_image)
