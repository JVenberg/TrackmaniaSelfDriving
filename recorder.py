import socket
import threading
import time
import inputs
import d3dshot
import json
import numpy as np
from inputs import devices
from PIL import Image

HOST = '127.0.0.1'
PORT = 65432


class XboxController(object):
    MAX_TRIG_VAL = 256
    MAX_JOY_VAL = 32768
    DEADZONE_CUTOFF = 0.1

    def __init__(self):
        if len(devices.gamepads) <= 0:
            raise ValueError('No Gamepad Detected')
        self.gamepad = devices.gamepads[0]

        self.LeftJoystickY = 0
        self.LeftJoystickX = 0
        self.RightJoystickY = 0
        self.RightJoystickX = 0
        self.LeftTrigger = 0
        self.RightTrigger = 0
        self.LeftBumper = 0
        self.RightBumper = 0
        self.A = 0
        self.X = 0
        self.Y = 0
        self.B = 0
        self.LeftThumb = 0
        self.RightThumb = 0
        self.Back = 0
        self.Start = 0
        self.LeftDPad = 0
        self.RightDPad = 0
        self.UpDPad = 0
        self.DownDPad = 0

        self._monitor_thread = threading.Thread(target=self._monitor_controller, args=(), daemon=True)
        self._monitor_thread.start()

    def stop(self):
        self._monitor_controller.stop()
    def read(self): # return the buttons/triggers that you care about in this method
        steer = self.LeftJoystickX if abs(self.LeftJoystickX) >= XboxController.DEADZONE_CUTOFF else 0
        acc = self.RightTrigger
        br = self.LeftTrigger
        pause = self.A
        start_stop = self.X
        return {
            "steering": steer,
            "acceleration": acc - br,
            "pause": pause,
            "start_stop": start_stop
        }


    def _monitor_controller(self):
        while True:
            try:
                events = self.gamepad.read()
            except inputs.UnpluggedError:
                self.gamepad = devices.gamepads[0]
                events = self.gamepad.read()

            for event in events:
                if event.code == 'ABS_Y':
                    self.LeftJoystickY = event.state / XboxController.MAX_JOY_VAL # normalize between -1 and 1
                elif event.code == 'ABS_X':
                    self.LeftJoystickX = event.state / XboxController.MAX_JOY_VAL # normalize between -1 and 1
                elif event.code == 'ABS_RY':
                    self.RightJoystickY = event.state / XboxController.MAX_JOY_VAL # normalize between -1 and 1
                elif event.code == 'ABS_RX':
                    self.RightJoystickX = event.state / XboxController.MAX_JOY_VAL # normalize between -1 and 1
                elif event.code == 'ABS_Z':
                    self.LeftTrigger = event.state / XboxController.MAX_TRIG_VAL # normalize between 0 and 1
                elif event.code == 'ABS_RZ':
                    self.RightTrigger = event.state / XboxController.MAX_TRIG_VAL # normalize between 0 and 1
                elif event.code == 'BTN_TL':
                    self.LeftBumper = event.state
                elif event.code == 'BTN_TR':
                    self.RightBumper = event.state
                elif event.code == 'BTN_SOUTH':
                    self.A = event.state
                elif event.code == 'BTN_NORTH':
                    self.Y = event.state
                elif event.code == 'BTN_WEST':
                    self.X = event.state
                elif event.code == 'BTN_EAST':
                    self.B = event.state
                elif event.code == 'BTN_THUMBL':
                    self.LeftThumb = event.state
                elif event.code == 'BTN_THUMBR':
                    self.RightThumb = event.state
                elif event.code == 'BTN_SELECT':
                    self.Back = event.state
                elif event.code == 'BTN_START':
                    self.Start = event.state
                elif event.code == 'BTN_TRIGGER_HAPPY1':
                    self.LeftDPad = event.state
                elif event.code == 'BTN_TRIGGER_HAPPY2':
                    self.RightDPad = event.state
                elif event.code == 'BTN_TRIGGER_HAPPY3':
                    self.UpDPad = event.state
                elif event.code == 'BTN_TRIGGER_HAPPY4':
                    self.DownDPad = event.state


if __name__ == '__main__':
    try:
        joy = XboxController()
        d = d3dshot.create()
        d.display = d.displays[1]
        width, height = d.display.resolution
        running = True
        start_stop_down = False
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
                            lines = conn_file.readlines()
                            last_line = lines[-1] if len(lines) > 0 else False
                            controller_data = joy.read()
                            
                            if controller_data['start_stop'] and not start_stop_down:
                                running = not running
                                start_stop_down = True
                            elif not controller_data['start_stop']:
                                start_stop_down = False

                            if last_line and running and not controller_data['pause']:
                                json_data = json.loads(last_line)
                                if json_data['speed'] is not None:
                                    img = d.screenshot(region=(0, 400, width, height - 460)).convert('L')
                                    img_width, img_height = img.size
                                    img \
                                        .transform((200, 200), Image.QUAD, (500, 0, 0, img_height, img_width, img_height, img_width - 500, 0)) \
                                        .save(r'D:/Documents/TrackmaniaSelfDrivingData/' + str(time.time()) + '.png')
                                    print([json_data['speed'], controller_data['steering'], controller_data['acceleration']])
                except BlockingIOError:
                    pass
                time.sleep(0.01)
    except KeyboardInterrupt:
        print('End')