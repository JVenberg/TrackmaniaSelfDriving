import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torch import nn
import torch.optim as optim
import numpy as np

import os
from functools import partial

from model import TrackmaniaNet
from dataset import TrackManiaDataset

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

import matplotlib.pyplot as plt


def load_data(data_dir='data'):
    training_data = TrackManiaDataset(data_dir, "train.csv", transform=transforms.Compose([
        transforms.ConvertImageDtype(torch.float)
    ]))
    test_data = TrackManiaDataset(data_dir, "test.csv", transform=transforms.Compose([
        transforms.ConvertImageDtype(torch.float)
    ]))

    train_dataloader = DataLoader(training_data, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(test_data, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

    return training_data, test_data


def accuracy_and_loss(net, dataloader, percent_err_thresh=0.1):
    loss_sum = 0.0
    correct = 0
    correct_speed = 0
    correct_steer = 0
    size = len(dataloader.dataset)
    criterion = nn.MSELoss()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with torch.no_grad():
        for batch in dataloader:
            images, labels = batch[0].to(device), batch[1].to(device)
            outputs = net(images)
            
            loss_sum += criterion(outputs, labels).item()
            
            per_err = torch.abs(torch.div(torch.sub(outputs, labels), labels))
            correct_speed += torch.sum(torch.le(per_err, percent_err_thresh).long()[:, 0])
            correct_steer += torch.sum(torch.le(per_err, percent_err_thresh).long()[:, 1])
            correct += torch.sum(torch.ge(torch.sum(torch.le(per_err, percent_err_thresh).long(), 1), 2).long())
    return correct / size, correct_speed / size, correct_steer / size,  loss_sum / size

# Created with help from: https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html
def train(config, checkpoint_dir=None, data_dir=None, epoch=10):
    trackmania_net = TrackmaniaNet(config['drop'])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trackmania_net.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(trackmania_net.parameters(), lr=config['lr'], momentum=0.9, weight_decay=config['decay'])

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        trackmania_net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    trainset, _ = load_data(data_dir)

    test_abs = int(len(trainset) * 0.8)
    train_subset, val_subset = random_split(
        trainset, [test_abs, len(trainset) - test_abs])

    trainloader = DataLoader(
        train_subset,
        batch_size=int(config['batch_size']),
        shuffle=True,
        num_workers=4)
    valloader = DataLoader(
        val_subset,
        batch_size=int(config['batch_size']),
        shuffle=True,
        num_workers=4)

    for epoch in range(epoch):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = trackmania_net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 100 == 99:  # print every 100 mini-batches
                print("[%d, %5d] loss: %.6f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0

        # Validation loss
        accuracy, speed_accuracy, steer_accuracy, val_loss = accuracy_and_loss(trackmania_net, valloader)

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((trackmania_net.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=val_loss, accuracy=accuracy, speed_accuracy=speed_accuracy, steer_accuracy=steer_accuracy)
    print("Finished Training")

def main(num_samples=10, max_num_epochs=10, gpus_per_trial=1):
    data_dir = os.path.abspath("./data")
    checkpoint_dir = os.path.abspath("./checkpoints")
    print(data_dir)
    config = {
        "drop": tune.quniform(0.15, 0.3, 0.025),
        "decay": tune.loguniform(5e-5, 5e-3),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([32, 64])
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "accuracy", "speed_accuracy", "steer_accuracy", "training_iteration"])
    result = tune.run(
        partial(train, data_dir=data_dir),
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        max_failures=5)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

    best_trained_model = TrackmaniaNet(best_trial.config['drop'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_trained_model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, _ = torch.load(os.path.join(best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    _, testset = load_data(data_dir)
    testloader = DataLoader(
        testset,
        batch_size=int(best_trial.config['batch_size']),
        shuffle=True,
        num_workers=4)
    test_acc, test_acc_speed, test_acc_steer, loss = accuracy_and_loss(best_trained_model, testloader)
    print("Best Trial Test Set\n\tAccuracy: {}\n\tSpeed Accuracy: {}\n\tSteer Accuracy: {}\n\tLoss: {}".format(test_acc, test_acc_speed, test_acc_steer, loss))

if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=100, max_num_epochs=10, gpus_per_trial=0.5)
