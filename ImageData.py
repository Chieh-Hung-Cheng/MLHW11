import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import os
import cv2
from PIL import Image

from Config import Config
from Utils import Utils


class ImageDataset:
    @staticmethod
    def get_trainset():
        train_transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Lambda(lambd=lambda x: cv2.Canny(np.array(x), 70, 130)),
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor()
        ])
        trainset = ImageFolder(root=os.path.join(Config.data_path, "train_data"), transform=train_transform)
        return trainset

    @staticmethod
    def get_testset():
        test_transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((32,32)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor()
        ])
        testset = ImageFolder(root=os.path.join(Config.data_path, "test_data"), transform=test_transform)
        return testset


if __name__ == "__main__":
    trainset = ImageDataset.get_trainset()
    pass
