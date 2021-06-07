import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch import nn
import torch.optim as optim
import numpy as np

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
        # transforms.Resize((32, 32))
    ]))
    test_data = TrackManiaDataset("data", "test.csv", transform=transforms.Compose([
        transforms.ConvertImageDtype(torch.float),
        # transforms.Resize((32, 32))
    ]))

    # view_data(training_data)

    train_dataloader = DataLoader(training_data, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(test_data, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

    # view_dataloader(train_dataloader)
    return train_dataloader, test_dataloader


def train(net, dataloader, epochs=1, lr=0.01, momentum=0.9, decay=0.0, verbose=True):
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
                    print("[%d, %5d] loss: %.6f" % (epoch + 1, i + 1, sum_loss / 100))
                sum_loss = 0.0
    return losses


def accuracy(net, dataloader):
    loss_sum = 0
    size = len(dataloader.dataset)
    criterion = nn.MSELoss()
    with torch.no_grad():
        for batch in dataloader:
            images, labels = batch[0].to(device), batch[1].to(device)
            outputs = net(images)
            loss_sum += criterion(outputs, labels).item()
    return loss_sum / size

def smooth(x, size):
    return np.convolve(x, np.ones(size) / size, mode="valid")

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))
    train_dataloader, test_dataloader = load_data()

    trackmania_net = TrackmaniaNet()
    conv_losses = train(trackmania_net, train_dataloader, epochs=15, lr=0.01)
    torch.save(trackmania_net.state_dict(), 'models/model_test.pth')
    plt.plot(smooth(conv_losses, 50))

    print("Training MSE loss: %f" % accuracy(trackmania_net, train_dataloader))
    print("Testing MSE loss: %f" % accuracy(trackmania_net, test_dataloader))

    plt.show()
    