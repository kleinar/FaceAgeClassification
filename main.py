from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import torch
import cv2
import yaml
from utils.model import AgeClassificator, FaceDetection
from utils.general import  create_folder_if_not_exists
from infer import inference
from PIL import Image
import io
import numpy as np


with open("config.yaml", "r") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

app = FastAPI()
face_detector = FaceDetection()
age_classificator = AgeClassificator(model_name=config['model'])
age_classificator.load_state_dict(torch.load(config['weights']))

fastapi_save = 'media/'

@app.post("/predict")
async def process_file(file: UploadFile = File(...)):
    # Считываем загруженный файл как изображение
    create_folder_if_not_exists(fastapi_save)
    content = await file.read()

    image = Image.open(io.BytesIO(content))
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(fastapi_save + file.filename, image)
    inference(fastapi_save+file.filename, mode='fastapi')
    #content = cv2.imread(fastapi_save+file.filename)

    return FileResponse(path='output/'+file.filename, filename=file.filename, media_type="image/png")