import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import torchvision.transforms as transforms
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from PIL import Image

import matplotlib.pyplot as plt
import csv
import os


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
        speed = float(row["speed"])
        steering = float(row["steering"])
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            speed = self.target_transform(speed)
            steering = self.target_transform(steering)
        return image, torch.from_numpy(np.array([speed, steering])).float()

    def __del__(self):
        self.anno_file.close()


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
    training_data = TrackManiaDataset("data", "train.csv", transform=transforms.Compose([
        transforms.ConvertImageDtype(torch.float),
        transforms.Resize((32, 32))
    ]))
    test_data = TrackManiaDataset("data", "test.csv", transform=transforms.Compose([
        transforms.ConvertImageDtype(torch.float),
        transforms.Resize((32, 32))

    ]))

    # view_data(training_data)

    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

    # view_dataloader(train_dataloader)
    return train_dataloader, test_dataloader


class ConvNet(nn.Module):
    def __init__(self):
        super(
            ConvNet, self
        ).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(1024, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x


def train(net, dataloader, epochs=1, lr=0.01, momentum=0.9, decay=0.0, verbose=1):
    net.to(device)
    losses = []
    criterion = nn.MSELoss()
    optimizer = optim.SGD(
        net.parameters(), lr=lr, momentum=momentum, weight_decay=decay
    )
    for epoch in range(epochs):
        sum_loss = 0.0
        for i, batch in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = batch[0].to(device), batch[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()  # autograd magic, computes all the partial derivatives
            optimizer.step()  # takes a step in gradient direction

            # print statistics
            losses.append(loss.item())
            sum_loss += loss.item()
            if i % 100 == 99:  # print every 100 mini-batches
                if verbose:
                    print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, sum_loss / 100))
                sum_loss = 0.0
        return losses


def accuracy(net, dataloader):
    loss_sum = 0
    total = 0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for batch in dataloader:
            images, labels = batch[0].to(device), batch[1].to(device)
            outputs = net(images)
            total += labels.size(0)
            loss_sum += torch.sqrt(criterion(outputs, labels))
    return loss_sum / total

def smooth(x, size):
    return np.convolve(x, np.ones(size) / size, mode="valid")


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))
train_dataloader, test_dataloader = load_data()

conv_net = ConvNet()
conv_losses = train(conv_net, train_dataloader, epochs=15, lr=.001)
plt.plot(smooth(conv_losses, 50))

print("Training accuracy: %f" % accuracy(conv_net, train_dataloader))
print("Testing  accuracy: %f" % accuracy(conv_net, test_dataloader))

plt.show()