import torch

import torchvision.transforms as transforms

from model import TrackmaniaNet

import d3dshot
import vgamepad as vg

from dataset import TrackManiaDataset
from torch.utils.data import DataLoader

import socket
import time
import json
import random

RESIZE = (64, 64)
SPEED_SCALE = 300
HOST = "127.0.0.1"
PORT = 65432


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    trackmania_net = TrackmaniaNet(drop_out=0.15)
    trackmania_net.load_state_dict(torch.load("models/model_large_tune.pth"))
    trackmania_net.eval()
    trackmania_net.to(device)

    d = d3dshot.create()
    d.display = d.displays[1]
    width, height = d.display.resolution

    to_tensor = transforms.ToTensor()

    gamepad = vg.VX360Gamepad()
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
                        print("Connected by", addr)
                        conn_file = conn.makefile("r")
                        smoothed_steering = 0
                        while True:
                            img = d.screenshot(region=(0, 400, width, height)).convert(
                                "L"
                            )
                            img_width, img_height = img.size
                            if RESIZE:
                                img = img.resize(RESIZE)

                            features = to_tensor(img).unsqueeze(0)

                            with torch.no_grad():
                                pred = trackmania_net(features.to(device))

                            # pred_speed = 120
                            pred_speed = pred[0][0] * SPEED_SCALE
                            pred_steering = pred[0][1]

                            # Driving angle smoothing taken from: https://github.com/SullyChen/Autopilot-TensorFlow/
                            smoothed_steering += (
                                0.2
                                * pow(
                                    abs((pred_steering - smoothed_steering)), 2.0 / 3.0
                                )
                                * (pred_steering - smoothed_steering)
                                / abs(pred_steering - smoothed_steering)
                            )

                            print(f"Speed: {pred_speed} Steering: {smoothed_steering}")

                            gamepad.left_joystick_float(
                                float(pred_steering), y_value_float=0.0
                            )

                            lines = conn_file.readlines()
                            last_line = lines[-1] if len(lines) > 0 else False
                            if last_line:
                                json_data = json.loads(last_line)
                                if json_data["speed"]:
                                    delta = pred_speed - json_data["speed"]
                                    if delta > 0:
                                        gamepad.left_trigger_float(
                                            value_float=random.uniform(-0.0001, 0.0001)
                                        )
                                        gamepad.right_trigger_float(
                                            value_float=random.uniform(0.89999, 0.90001)
                                        )
                                    elif delta < -30:
                                        gamepad.left_trigger_float(
                                            value_float=random.uniform(-0.0001, 0.0001)
                                        )
                                        gamepad.right_trigger_float(
                                            value_float=random.uniform(-0.0001, 0.0001)
                                        )
                            gamepad.update()
                except BlockingIOError:
                    pass
                time.sleep(0.01)
    except KeyboardInterrupt:
        print("Done")
