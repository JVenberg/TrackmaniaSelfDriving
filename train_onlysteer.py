import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch import nn
import torch.optim as optim
import numpy as np

from model import TrackmaniaNetOnlySteer
from dataset import TrackManiaDataset

import matplotlib.pyplot as plt


def load_data():
    training_data = TrackManiaDataset("data", "train.csv", only_steer=True, transform=transforms.Compose([
        transforms.ConvertImageDtype(torch.float)
    ]))
    test_data = TrackManiaDataset("data", "test.csv", only_steer=True, transform=transforms.Compose([
        transforms.ConvertImageDtype(torch.float)
    ]))

    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

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


def accuracy_and_loss(net, dataloader, percent_err_thresh=0.1):
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
            
            per_err = torch.abs(torch.div(torch.sub(outputs, labels), labels))
            correct_steer += torch.sum(torch.le(per_err, percent_err_thresh).long()[:, 0])
            correct += torch.sum(torch.ge(torch.sum(torch.le(per_err, percent_err_thresh).long(), 1), 1).long())
    return correct / size * 100, correct_steer / size * 100,  loss_sum / size
    

def smooth(x, size):
    return np.convolve(x, np.ones(size) / size, mode="valid")

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))
    train_dataloader, test_dataloader = load_data()

    trackmania_net = TrackmaniaNetOnlySteer(drop_out=0.25)
    conv_losses = train(trackmania_net, train_dataloader, epochs=15, lr=0.002024490771869695, decay=0.004213432606306408)
    torch.save(trackmania_net.state_dict(), 'models/model.pth')
    plt.plot(smooth(conv_losses, 50))

    print("Training accuracy & MSE loss: %f%%, %f%%, %f" % accuracy_and_loss(trackmania_net, train_dataloader))
    print("Testing accuracy & MSE loss: %f%%, %f%%, %f" % accuracy_and_loss(trackmania_net, test_dataloader))

    plt.show()
    