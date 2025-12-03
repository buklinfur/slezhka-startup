"""
Модуль для детекции лиц и определения направления взгляда
"""

import cv2
import torch

import numpy as np
import torch.nn.functional as F

from ultralytics import YOLO
from torchvision import transforms

import utils.helpers


def pre_process(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std = 0.5)
    ])

    image = transform(image)
    image_batch = image.unsqueeze(0)
    return image_batch


class yolo_face:
    """
    использует yolov8 с кастомными весами для детекции лиц на фото
    """
    def __init__(self, device):
        model_path = 'weights/yolov8face.pt'
        self.model = YOLO(model_path).to(device)


    def forward(self, img):
        """
        принимает обработанную картинку в виде pytoch 'device' tensor [batch, 3, 640, 640]

        возвращает список вещественных координат bbox детекций
        """
        with torch.no_grad():
            bboxes = self.model(img)[0].cpu()
        bbox_positions = [bbox.boxes.xyxyn[0] for bbox in bboxes]
        return bbox_positions
    
    
    def make_face_batch(self, image, bbox_list, preprocessor):
        """
        Создает batch из обрезанных лиц с изображения по списку bbox с yolo 
    
        image -> необработанное cv2 изображение в исходном разрешении
        bbox_list -> список bbox для этого изображения, который возвращает метод forward
        preprocessor -> функция для предобработки изображения для дальнейшей передачи в mobile_gaze

        возвращает pytorch tensor [faces, 3, 640, 640] (с текущей функцией обработки)
        """
        face_crops = []
        for bbox in bbox_list:
            bbox_cords = bbox
            bbox_cords[[0, 2]] *= image.shape[1]
            bbox_cords[[1, 3]] *= image.shape[0]
            x_min, y_min, x_max, y_max = map(int, bbox_cords)
            crop = image[y_min:y_max, x_min:x_max]
            crop = preprocessor(crop)
            face_crops.append(crop)
        return torch.concatenate(face_crops)


class mobile_gaze:
    """
    использует mobile gaze для оценки направления взгляда
    """
    def __init__(self, device, model='resnet18', weight='weights/resnet18.pt', bins = 90, binwidth = 4, angle = 180):
        self.model = utils.helpers.get_model(model, bins, inference_mode=True)
        state_dict = torch.load(weight, map_location=device)
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()

        self.idx_tensor = torch.arange(bins, device=device, dtype=torch.float32)
        self.binwidth = binwidth
        self.angle = angle

    def forward(self, face_img):
        """
        face_img -> batch предобработанных обрезанных изображений лиц pytorch tensor [batch, 3, 640, 640]

        возвращает наклоны (тангаж и рыскание) вектора взгляда
        """
        with torch.no_grad():
            pitch, yaw = self.model(face_img)

            pitch_predicted, yaw_predicted = F.softmax(pitch, dim=1), F.softmax(yaw, dim=1)
            pitch_predicted = torch.sum(pitch_predicted * self.idx_tensor, dim=1) * self.binwidth - self.angle
            yaw_predicted = torch.sum(yaw_predicted * self.idx_tensor, dim=1) * self.binwidth - self.angle

            pitch_predicted_np = pitch_predicted.cpu().numpy()
            yaw_predicted_np = yaw_predicted.cpu().numpy()

        pitch_predicted = np.radians(pitch_predicted_np)
        yaw_predicted = np.radians(yaw_predicted_np)

        return pitch_predicted, yaw_predicted

    def draw_result(self, img, pitch, yaw, bbox_list):
        """
        рисует результат работы на изображении, изменяя img в процессе

        img -> необработанное cv2 изображение в исходном разрешении
        pitch -> тангаж вектора взгляда
        yaw -> рыскание вектора взгляда
        bbox_list -> список bbox для этого изображения, который возвращает метод forward класса yolo_face
        """
        for i in range(len(bbox_list)):
            utils.helpers.draw_bbox_gaze(img, bbox_list[i], pitch[i], yaw[i])


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('using ' + device)

    unprocessed = cv2.imread('examples/image.png')
    img = pre_process(unprocessed).to(device)
  
    face_detector = yolo_face(device)
    gaze_detector = mobile_gaze(device)

    bbox_list = face_detector.forward(img)
    face_batch = face_detector.make_face_batch(unprocessed, bbox_list, pre_process)
    pitch, yaw = gaze_detector.forward(face_batch.to(device))
    gaze_detector.draw_result(unprocessed, pitch, yaw, bbox_list)

    cv2.imwrite('checkmeout.png', unprocessed)