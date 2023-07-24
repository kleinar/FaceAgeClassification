import argparse
import cv2
import os
import numpy as np
import torch

from utils.model import AgeClassificator, FaceDetection
from utils.datasets import img_to_tensor
from utils.general import  visualize_image

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', nargs='+', type=str, default='resnet18', help='pretrained model name')
    parser.add_argument('--weights', nargs='+', type=str, default='models/best_model.pt', help='model path')
    parser.add_argument('--source', type=str, default= '1', help='file/dir/video')
    parser.add_argument('--imgsz', type=list, default=224, help='224 or 512')
    parser.add_argument('--save_path', nargs='+', type=str, default='output/', help='save results path')

    opt = parser.parse_args()
    return opt

def main(opt):
    face_detector = FaceDetection()
    age_classificator = AgeClassificator(model_name=opt.model)
    age_classificator.load_state_dict(torch.load(opt.weights))

    if opt.source.endswith(('.png', '.jpg')):
        img = cv2.imread(opt.source)
        img = infer_image(img, face_detector, age_classificator)
        cv2.imwrite(opt.save_path + opt.source, img)
    elif opt.source.endswith(('.avi', '.mp4')):
        cap = cv2.VideoCapture(opt.source)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))

        size = (frame_width, frame_height)
        result = cv2.VideoWriter(opt.save_path + opt.source,
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

    elif opt.source.endswith(''):
        image_list = os.listdir(opt.source)
        for img_name in image_list:
            img = cv2.imread(opt.source + img_name)
            img = infer_image(img, face_detector, age_classificator)
            cv2.imwrite(opt.save_path + img_name, img)


def infer_image(img:np.array, face_detector: FaceDetection, age_classificator: AgeClassificator):
    faces = face_detector.detect(img)
    for (x, y, w, h) in faces:
        face = img[y:y + h, x:x + w]
        face = img_to_tensor(face, opt.imgsz)
        face_age = age_classificator(face).cpu().detach().numpy()[0]
        img = visualize_image(img, (x, y, x + w, y + h), face_age)
    return img


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)