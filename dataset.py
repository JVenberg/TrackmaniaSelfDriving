
import csv
import os

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

import numpy as np

class TrackManiaDataset(Dataset):
    def __init__(
        self, data_dir, annotations_file_name, transform=None, target_transform=None
    ):
        self.anno_file = open(os.path.join(data_dir, annotations_file_name))
        self.img_labels = list(csv.DictReader(self.anno_file))
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        row = self.img_labels[idx]
        img_path = os.path.join(self.data_dir, row["img_file"])
        image = read_image(img_path)
        speed = float(row["speed"]) / 400
        steering = float(row["steering"])
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            speed = self.target_transform(speed)
            steering = self.target_transform(steering)
        return image, torch.from_numpy(np.array([speed, steering])).float()

    def __del__(self):
        self.anno_file.close()
        