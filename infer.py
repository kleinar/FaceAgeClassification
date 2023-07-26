import argparse
import cv2
import os
import numpy as np
import torch
import yaml
from utils.model import AgeClassificator, FaceDetection
from utils.general import visualize_image, img_to_tensor, create_folder_if_not_exists


with open("config.yaml", "r") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

face_detector = FaceDetection()
age_classificator = AgeClassificator(model_name=config['model'])
age_classificator.load_state_dict(torch.load(config['weights']))

def inference(source:str = '', mode:str = 'infer'):
    if mode == 'infer':
        source = config['source-path']
    create_folder_if_not_exists(config['save-results-path'])
    if source.endswith(('.png', '.jpg')):
        img = cv2.imread(source)
        img = infer_image(img, face_detector, age_classificator)
        source = source.split('/')[-1]
        cv2.imwrite(config['save-results-path'] + source, img)
    elif source.endswith(('.avi', '.mp4')):
        cap = cv2.VideoCapture(config['source-path'])
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))

        size = (frame_width, frame_height)
        result = cv2.VideoWriter(config['save-results-path'] + source,
                                 cv2.VideoWriter_fourcc(*'MJPG'),
                                 10, size)
        if not cap.isOpened():
            print("Cannot open video")
            exit()
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            frame = infer_image(frame, face_detector, age_classificator)
            result.write(frame)

        cap.release()
        result.release()

    elif source.endswith(''):
        image_list = os.listdir(source)
        for img_name in image_list:
            img = cv2.imread(source + img_name)
            img = infer_image(img, face_detector, age_classificator)
            cv2.imwrite(config['save-results-path'] + img_name, img)


def infer_image(img:np.array, face_detector: FaceDetection, age_classificator: AgeClassificator):
    faces = face_detector.detect(img)
    for (x, y, w, h) in faces:
        face = img[y:y + h, x:x + w]
        face = img_to_tensor(face, config['img-size'])
        face_age = age_classificator(face).cpu().detach().numpy()[0]
        img = visualize_image(img, (x, y, x + w, y + h), face_age)
    return img


if __name__ == '__main__':
    inference()