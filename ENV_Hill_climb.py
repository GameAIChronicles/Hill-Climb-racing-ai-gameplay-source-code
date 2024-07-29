import os
import tensorflow
import keras
import mss
import time
import numpy as np
from gymnasium.spaces import Box, Discrete
import pydirectinput
import pyautogui
import cv2 as cv
from gymnasium import Env


class ENV_hill_climb(Env):
    def __init__(self):
        self.observation_space = Box(low=0, high=255, shape=(1, 350, 100), dtype=np.uint8)  # Observation
        # 0 = Backward, 1 = forward, 2 = nothing
        self.action_space = Discrete(3)  # Action
        self.game_location = {'top': 250, 'left': 0, 'width': 1050, 'height': 300}
        self.done_loc = {'top': 450, 'left': 400, 'width': 200, 'height': 100}
        self.cap = mss.mss()
        self.model = keras.models.load_model('data/models/GameOverModel/GameOver.h5')
        self.coin_img = cv.cvtColor(np.array(self.cap.grab({'top': 70, 'left': 70, 'width': 280, 'height': 60}))[:, :, :3], cv.COLOR_BGR2GRAY)
        self.distance = cv.cvtColor(np.array(self.cap.grab({'top': 40, 'left': 330, 'width': 280, 'height': 60}))[:, :, :3], cv.COLOR_BGR2GRAY)
        self.diamond = cv.cvtColor(np.array(self.cap.grab({'top': 130, 'left': 70, 'width': 200, 'height': 60}))[:, :, :3], cv.COLOR_BGR2GRAY)

    #Taking an action in the Env
    def step(self, action):
        Action_map = {0: 'left', 1: 'right', 2: 'nothing'}
        if action == 1:
            pyautogui.keyDown(Action_map[action])
            time.sleep(0.8)
            pyautogui.keyUp(Action_map[action])
        if action == 0:
            pyautogui.keyDown(Action_map[action])
            time.sleep(0.4)
            pyautogui.keyUp(Action_map[action])
        if action == 2:
            pass
        reward = 0

        coin_img2 = cv.cvtColor(np.array(self.cap.grab({'top': 70, 'left': 70, 'width': 280, 'height': 60}))[:, :, :3], cv.COLOR_BGR2GRAY)
        cm1 = cv.compare(self.coin_img, coin_img2, 0)
        if cm1.all():
            reward += 0
        else:
            reward += 1
            self.coin_img = coin_img2

        distance2 = cv.cvtColor(np.array(self.cap.grab({'top': 40, 'left': 330, 'width': 280, 'height': 60}))[:, :, :3], cv.COLOR_BGR2GRAY)
        cm2 = cv.compare(self.distance, distance2, 0)
        if cm2.all():
            reward += 0
        else:
            reward += 1
            self.distance = distance2

        diamond2 = cv.cvtColor(np.array(self.cap.grab({'top': 130, 'left': 70, 'width': 200, 'height': 60}))[:, :, :3], cv.COLOR_BGR2GRAY)
        cm3 = cv.compare(self.diamond, diamond2, 0)
        if cm3.all():
            reward += 0
        else:
            reward += 1
            self.diamond = diamond2

        done, done_cap = self.GAME_over()
        if done:
            reward -= 50
        new_obs = self.get_obs()

        return new_obs, reward, done, False, {}

    #Reseting the Env
    def render(self):
        pass

    #Reseting the Env
    def reset(self, **kwargs):
        pydirectinput.click(x=1000, y=750)
        time.sleep(1.5)
        pydirectinput.press('space')
        time.sleep(0.5)
        pydirectinput.press('Enter')
        time.sleep(0.5)
        self.coin_img = cv.cvtColor(np.array(self.cap.grab({'top': 70, 'left': 70, 'width': 280, 'height': 60}))[:, :, :3], cv.COLOR_BGR2GRAY)
        self.distance = cv.cvtColor(np.array(self.cap.grab({'top': 40, 'left': 330, 'width': 280, 'height': 60}))[:, :, :3], cv.COLOR_BGR2GRAY)
        self.diamond = cv.cvtColor(np.array(self.cap.grab({'top': 130, 'left': 70, 'width': 200, 'height': 60}))[:, :, :3], cv.COLOR_BGR2GRAY)
        return self.get_obs(), {}

    #The image of the Agent in the Game
    def get_obs(self):
        raw = np.array(self.cap.grab(self.game_location))[:, :, :3]
        gray = cv.cvtColor(raw, cv.COLOR_BGR2GRAY)
        resized = cv.resize(gray, (350, 100))
        channel = np.reshape(resized, (1, 350, 100))
        return channel

    # Game over Indication
    def GAME_over(self):
        Done_cap = np.array(self.cap.grab(self.done_loc))[:, :, :3]
        List_data_img = [x for x in os.listdir(os.path.join('data', 'img'))]
        img = tensorflow.image.resize(Done_cap, (100, 100))
        pre = self.model.predict(np.expand_dims(img / 225, 0), verbose=0)
        DONE = False
        if List_data_img[pre.argmax()] == 'done_img':
            DONE = True
        return DONE, Done_cap

    #Closeing the window
    def close(self):
        pass


if '__main__' == __name__:
    print('hi')