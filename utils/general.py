import numpy as np
import cv2
import torch
from torch.autograd import Variable
from albumentations.pytorch import ToTensor
import albumentations as A
from utils.model import AgeClassificator
import os

def visualize_image(img:np.array, face_coord:tuple, age:float):
    '''

    :param img: numpy array format image
    :param face_coord: (xmin, ymin,xmax, ymax)
    :param age: human age
    :return: visualize image with rectangles araund face and put text age of human

    '''
    start_point = (face_coord[0], face_coord[1])
    end_point = (face_coord[2], face_coord[3])
    # Blue color in BGR
    color = (255, 0, 0)
    # Line thickness of 2 px
    thickness = 2
    fontScale = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Using cv2.rectangle() method
    # Draw a rectangle with blue line borders of thickness of 2 px
    img = cv2.rectangle(img, start_point, end_point, color, thickness)
    img = cv2.putText(img, str(age), start_point, font,
                        fontScale, color, thickness, cv2.LINE_AA)

    return img

def calculate_metric(model, loader, device, metric):
    '''

    :param model: Face age classification model
    :param loader: Dataloader
    :param device: cpu or cuda
    :param metric: metric function
    :return: metric value as torch Tensor
    '''
    metric_value = 0
    with torch.no_grad():
        for data in loader:
            inputs = Variable(data['image']).to(device)
            labels = Variable(data['age']).to(device)

            output = model(inputs).view(len(inputs))

            metric_value += metric(output, labels)
    return metric_value

def img_to_tensor(img:np.array, img_size:int):
    '''

    :param img: cropped face image
    :param img_size: image size h or w
    :return: converted numpy array to tensor with format (1, c, h, w)
    '''
    test_transforms = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    result = test_transforms(image = img)
    result = ToTensor()(image=result['image'])['image']
    return torch.unsqueeze(result, 0)

def get_age_from_face(img:np.array, face: list, age_classificator: AgeClassificator, imgsz:int):
    x, y, h, w = face
    face_img = img[y:y + h, x:x + w]
    face_img = img_to_tensor(face_img, imgsz)
    face_age = age_classificator(face_img).cpu().detach().numpy()[0]
    return face_age

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print("Папка успешно создана")
    else:
        print("Папка уже существует")
