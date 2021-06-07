import torch

import torchvision.transforms as transforms

from model import TrackmaniaNet

import d3dshot
import matplotlib.pyplot as plt
import gamepyd

from dataset import TrackManiaDataset
from torch.utils.data import DataLoader

import socket
import time
import json

RESIZE = (32, 32)
SPEED_SCALE = 200
HOST = '127.0.0.1'
PORT = 65432

# def view_data(data):
#     figure = plt.figure(figsize=(8, 8))
#     cols, rows = 3, 3
#     for i in range(1, cols * rows + 1):
#         sample_idx = torch.randint(len(data), size=(1,)).item()
#         img, (speed, steering) = data[sample_idx]
#         figure.add_subplot(rows, cols, i)
#         plt.title("Speed: " + str(speed) + "\nSteering: " + str(steering))
#         plt.axis("off")
#         plt.imshow(img.squeeze(), cmap="gray")
#     plt.show()


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

    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True, pin_memory=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True, pin_memory=True)

    # view_dataloader(train_dataloader)
    return train_dataloader, test_dataloader

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    trackmania_net = TrackmaniaNet()
    trackmania_net.load_state_dict(torch.load('models/model_nvidia_atan.pth'))
    trackmania_net.eval()
    trackmania_net.to(device)

    d = d3dshot.create()
    d.display = d.displays[1]
    width, height = d.display.resolution

    to_tensor = transforms.ToTensor()

    train_dataloader, test_dataloader = load_data()

    features, labels = next(iter(test_dataloader))

    pad = gamepyd.wPad()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((HOST, PORT))
            s.listen()
            s.setblocking(False)
            while True:
                try:
                    conn, addr = s.accept()
                    with conn:
                        conn.setblocking(False)
                        print('Connected by', addr)
                        conn_file = conn.makefile('r')
                        while True:
                            img = d.screenshot(region=(0, 400, width, height)).convert('L')
                            img_width, img_height = img.size
                            if RESIZE:
                                img = img.resize(RESIZE)
                            
                            features = to_tensor(img).unsqueeze(0)

                            with torch.no_grad():
                                pred = trackmania_net(features.to(device))
                                
                            # pred_speed = 120
                            pred_speed = pred[0][0] * SPEED_SCALE
                            pred_steering = pred[0][1]

                            # print(f"Speed: {pred_speed} Steering: {pred_steering}")
                            
                            controls = {
                                'Lx': float(pred_steering),
                                # 'LT': 0,
                                # 'RT': 0
                            }

                            lines = conn_file.readlines()
                            last_line = lines[-1] if len(lines) > 0 else False
                            if last_line:
                                json_data = json.loads(last_line)
                                if json_data['speed']:
                                    delta = pred_speed - json_data['speed']
                                    if delta > 0:
                                        controls['LT'] = 0
                                        controls['RT'] = 0.9
                                    elif delta < -30:
                                        controls['LT'] = 0
                                        controls['RT'] = 0
                            pad.playMoment(controls, check=False)
                except BlockingIOError:
                    pass
                time.sleep(0.01)
    except KeyboardInterrupt:
        pad.UnPlug()