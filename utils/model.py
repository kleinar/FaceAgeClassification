import torch
import pretrainedmodels
import torch.nn as nn
import cv2
import numpy as np

class AgeClassificator(nn.Module):
    '''
    Human age determination model

    input initialization: model name - 'resnet18', 'resnet34', etc
                          pretrained - 'imagenet'

    input forward: image batch format (batch_size, h, w, channel) in torch.FloatTensor
    output forward: model result in torch.FloatTensor

    '''

    def __init__(self, model_name: str = 'resnet18', pretrained: str = "imagenet"):
        super(AgeClassificator, self).__init__()
        self.model = pretrainedmodels.__dict__[model_name](pretrained=pretrained)
        self.model.last_linear = nn.Linear(self.model.last_linear.in_features, 1)

    def forward(self, x: torch.FloatTensor):
        return self.model(x)


class FaceDetection(nn.Module):
    '''
    Human face detection with haarcascade. If you want to change face detector, change self.detector

    input: image in numpy array format
    output: list of detected faces

    '''

    def __init__(self):
        super(FaceDetection, self).__init__()
        self.detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def detect(self, image: np.array):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face = self.detector.detectMultiScale(
            gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )
        return face

