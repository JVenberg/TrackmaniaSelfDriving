import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch import nn
import torch.optim as optim
import numpy as np

from model import TrackmaniaNet
from dataset import TrackManiaDataset

import matplotlib.pyplot as plt

# LEARNING_RATE = 0.048749
# DECAY = 0.00013067
# BATCH_SIZE = 64
# DROP_OUT = 0.175

LEARNING_RATE = 0.025557158229886558
DECAY = 0.0004193392285214784
BATCH_SIZE = 64
DROP_OUT = 0.15


def load_data(batch_size=64):
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
            inputs, labels = batch[0].to(device), batch[1].to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            sum_loss += loss.item()
            if i % 100 == 99:
                if verbose:
                    print("[%d, %5d] loss: %.6f" % (epoch + 1, i + 1, sum_loss / 100))
                sum_loss = 0.0
    return losses


def accuracy_and_loss(net, dataloader, err_thresh=0.1):
    loss_sum = 0.0
    correct = 0
    correct_speed = 0
    correct_steer = 0
    size = len(dataloader.dataset)
    criterion = nn.MSELoss()

    with torch.no_grad():
        for batch in dataloader:
            images, labels = batch[0].to(device), batch[1].to(device)
            outputs = net(images)

            loss_sum += criterion(outputs, labels).item()

            err = torch.abs(torch.sub(outputs, labels))
            correct_speed += torch.sum(torch.le(err, err_thresh).long()[:, 0])
            correct_steer += torch.sum(torch.le(err, err_thresh).long()[:, 1])
            correct += torch.sum(
                torch.ge(torch.sum(torch.le(err, err_thresh).long(), 1), 2).long()
            )
    return (
        correct / size * 100,
        correct_speed / size * 100,
        correct_steer / size * 100,
        loss_sum / size,
    )


def smooth(x, size):
    return np.convolve(x, np.ones(size) / size, mode="valid")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))
    train_dataloader, test_dataloader = load_data(BATCH_SIZE)

    trackmania_net = TrackmaniaNet(drop_out=DROP_OUT)
    conv_losses = train(
        trackmania_net, train_dataloader, epochs=15, lr=LEARNING_RATE, decay=DECAY
    )
    torch.save(trackmania_net.state_dict(), "models/model_large_tune.pth")
    plt.title('Loss vs Num Batches')
    plt.ylabel('Loss')
    plt.xlabel('Num Batches')
    plt.plot(smooth(conv_losses, 50))

    print(
        "Training accuracy & MSE loss: %f%%, %f%%, %f%%, %f"
        % accuracy_and_loss(trackmania_net, train_dataloader)
    )
    print(
        "Testing accuracy & MSE loss: %f%%, %f%%, %f%%, %f"
        % accuracy_and_loss(trackmania_net, test_dataloader)
    )

    plt.show()
