import os.path

import mss
import numpy as np
import cv2 as cv
import time

time.sleep(3)
cap = mss.mss()

#game_location = {'top': 250, 'left': 0, 'width': 1050, 'height': 300}
done_loc = {'top': 450, 'left': 400, 'width': 200, 'height': 100}
Done_cap = np.array(cap.grab(done_loc))[:, :, :3]
print(Done_cap)
gray = cv.cvtColor(Done_cap, cv.COLOR_BGR2GRAY)
resized = cv.resize(gray, (100, 100))
for n in range(100):
    cv.imwrite(filename=os.path.join('data', 'img', 'not_done_img', f'nd{n}.jpg'), img=resized)