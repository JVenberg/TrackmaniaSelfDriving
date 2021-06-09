import csv
import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt

SPEED_SCALE = 400


class TrackManiaDataset(Dataset):
    def __init__(
        self,
        data_dir,
        annotations_file_name,
        only_steer=False,
        transform=None,
        target_transform=None,
    ):
        anno_file = open(os.path.join(data_dir, annotations_file_name))
        self.img_labels = list(csv.DictReader(anno_file))
        anno_file.close()
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        self.only_steer = only_steer

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        row = self.img_labels[idx]
        img_path = os.path.join(self.data_dir, row["img_file"])
        image = read_image(img_path)
        speed = float(row["speed"]) / SPEED_SCALE
        steering = float(row["steering"])
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            speed = self.target_transform(speed)
            steering = self.target_transform(steering)

        if self.only_steer:
            return image, torch.from_numpy(np.array([steering])).float()

        return image, torch.from_numpy(np.array([speed, steering])).float()


def view_data(data):
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(data), size=(1,)).item()
        img, speed, steering = data[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title("Speed: " + str(speed) + "\nSteering: " + str(steering))
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()


def view_dataloader(dataloader):
    train_features, train_speed, train_steering = next(iter(dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Speed batch shape: {train_speed.size()}")
    print(f"Steering batch shape: {train_steering.size()}")
    img = train_features[0].squeeze()
    speed = train_speed[0]
    steering = train_steering[0]
    plt.imshow(img, cmap="gray")
    print(f"Speed: {speed} Steering: {steering}")
    plt.show()


def load_data():
    training_data = TrackManiaDataset(
        "data",
        "train.csv",
        transform=transforms.Compose([transforms.ConvertImageDtype(torch.float)]),
    )
    test_data = TrackManiaDataset(
        "data",
        "test.csv",
        transform=transforms.Compose([transforms.ConvertImageDtype(torch.float)]),
    )

    return training_data, test_data


def load_dataloaders(training_data, test_data, batch_size=64):
    train_dataloader = DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    return train_dataloader, test_dataloader


if __name__ == "__main__":
    training_data, test_data = load_data()
    view_data(training_data)

    training_dataloader, _ = load_dataloaders(training_data, test_data)
    view_dataloader(training_dataloader)