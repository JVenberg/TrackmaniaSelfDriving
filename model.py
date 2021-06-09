import torch
from torch import nn
import torch.nn.functional as F


# Network based on NVidia paper: https://arxiv.org/pdf/1604.07316.pdf
class TrackmaniaNet(nn.Module):
    def __init__(self, drop_out=0.2):
        super(TrackmaniaNet, self).__init__()
        self.drop_out = drop_out
        self.conv1 = nn.Conv2d(1, 24, 5, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(24, 36, 5, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(36)
        self.conv3 = nn.Conv2d(36, 48, 5, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(48)
        self.conv4 = nn.Conv2d(48, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(64)

        self.flat = nn.Flatten(1)
        self.fc1 = nn.Linear(3136, 1164)
        self.fc2 = nn.Linear(1164, 100)
        self.fc3 = nn.Linear(100, 50)
        self.fc4 = nn.Linear(50, 10)
        self.fc5 = nn.Linear(10, 2)

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
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)

        x = self.flat(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, self.drop_out)
        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x, self.drop_out)
        x = self.fc3(x)
        x = F.relu(x)
        x = F.dropout(x, self.drop_out)
        x = self.fc4(x)
        x = F.relu(x)
        x = F.dropout(x, self.drop_out)
        x = self.fc5(x)
        x = torch.atan(x)
        return x


class TrackmaniaNetOnlySteer(nn.Module):
    def __init__(self, drop_out=0.2):
        super(TrackmaniaNetOnlySteer, self).__init__()
        self.drop_out = drop_out
        self.conv1 = nn.Conv2d(1, 24, 5, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(24, 36, 5, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(36)
        self.conv3 = nn.Conv2d(36, 48, 5, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(48)
        self.conv4 = nn.Conv2d(48, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(64)

        self.flat = nn.Flatten(1)
        self.fc1 = nn.Linear(576, 1164)
        self.fc2 = nn.Linear(1164, 100)
        self.fc3 = nn.Linear(100, 50)
        self.fc4 = nn.Linear(50, 10)
        self.fc5 = nn.Linear(10, 1)

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
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)

        x = self.flat(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, self.drop_out)
        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x, self.drop_out)
        x = self.fc3(x)
        x = F.relu(x)
        x = F.dropout(x, self.drop_out)
        x = self.fc4(x)
        x = F.relu(x)
        x = F.dropout(x, self.drop_out)
        x = self.fc5(x)
        x = torch.atan(x)
        return x
