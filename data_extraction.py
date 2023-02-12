import cv2
import numpy as np
import pandas as pd
import random

chessboard_size = (9, 12)

wc = np.zeros((108, 3), np.float32)
for index, val in enumerate(range(5,-1, -1)):
    wc[9*index:9*(index+1), :] = 24*np.array([val*np.ones(9), np.zeros(9), range(1,10)]).T
for index, val in enumerate(range(1,7)):
    index = index+6;
    wc[9*index:9*(index+1), :] = 24*np.array([np.zeros(9), val*np.ones(9), range(1,10)]).T


no_images=15
samples=16
for imge in range(no_images):
    img = cv2.imread("image1.jpg")
    img = cv2.resize(img, None, fx=0.2, fy=0.2)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    if found:
        corners=5*corners
        train_index = random.sample(range(108), 16)
        corners=corners.reshape((108,2))
        coordinates = np.hstack((wc[train_index],corners[train_index]))
        df = pd.DataFrame(coordinates,columns=['x','y','z','X_img','Y_img'])
        df.to_excel("corners"+str(imge)+".xlsx",index=False)