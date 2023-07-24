import cv2
import albumentations as A
import os

from albumentations.pytorch import ToTensor
from torch.utils.data import DataLoader


class AgeDataset(DataLoader):
    '''
    Класс для работы с датасетом определения возраста
    '''

    def __init__(self, list_dir: str, transform: A.Compose, path: str):
        self.list_dir = list_dir
        self.transform = transform
        self.path = path

    def __getitem__(self, index: int):
        img_name = self.list_dir[index]
        age = int(img_name.split('_')[0])
        img_path = os.path.join(self.path, img_name)
        img = cv2.imread(img_path)

        result = self.transform(
            image=img,
        )
        result = {
            'image': ToTensor()(image=result['image'])['image'],
            'age': age,
        }
        return result

    def __len__(self, ):
        return len(self.list_dir)

def train_val_test_dataloader(dataset_path:str = '', img_size:int = 224, batch_size:int = 32,
                                                    train_list:list = [], val_list:list = [], test_list:list =[]):

    '''
    Function where we define dataloaders for training, validation and test

    :param dataset_path: path to dataset
    :param img_size: input image size
    :param batch_size: batch size
    :param train_list: list of images name for train
    :param val_list: list of images name for validation
    :param test_list: list of images name for test

    :return: train, validation and test Dataloaders
    '''

    #Train Dataloaders
    train_transforms = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    train_dataset = AgeDataset(
        list_dir=train_list,
        path=dataset_path,
        transform=train_transforms
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )


    #Validation Dataloaders
    val_transforms = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    val_dataset = AgeDataset(
        list_dir=val_list,
        path=dataset_path,
        transform=val_transforms
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    #Test Dataloaders
    test_transforms = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    test_dataset = AgeDataset(
        list_dir=test_list,
        path=dataset_path,
        transform=test_transforms
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader
