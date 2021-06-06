import torch

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from model import TrackmaniaNet
from dataset import TrackManiaDataset


import matplotlib.pyplot as plt

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
    train_features, train_labels = next(iter(dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[0].squeeze()
    speed = train_labels[0][0]
    steering = train_labels[0][1]



    plt.title("Speed: " + str(speed) + "\nSteering: " + str(steering))
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

trackmania_net = TrackmaniaNet()
trackmania_net.load_state_dict(torch.load('model.pth'))

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

trackmania_net.to(device)

train_dataloader, test_dataloader = load_data()

features, labels = next(iter(test_dataloader))

print(f"Feature batch shape: {features.size()}")
print(f"Labels batch shape: {labels.size()}")
img = features[0].squeeze()
actual_speed = labels[0][0]
actual_steering = labels[0][1]

with torch.no_grad():
    pred = trackmania_net(features.to(device))
    
pred_speed = pred[0][0]
pred_steering = pred[0][1]

plt.title("Act Speed: " + str(actual_speed) +
    "\nAct Steering: " + str(actual_steering) +
    "\nPred Speed: " + str(pred_speed) +
    "\nPred Steering: " + str(pred_steering))
plt.imshow(img, cmap="gray")
print(f"Act Speed: {actual_speed} Act Steering: {actual_steering}")
plt.show()