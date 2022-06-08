import cv2 as cv2
import numpy as np
import statistics
   
# Create a VideoCapture object and read from input file
count = 0
cap = cv2.VideoCapture("SampleVideo4.mp4")
success, image = cap.read()
features = []
success = True
width = cap.get(cv2. CAP_PROP_FRAME_WIDTH )
height = cap.get(cv2. CAP_PROP_FRAME_HEIGHT )






while success:

    cap.set(cv2.CAP_PROP_POS_MSEC, (count * 1000))  # added this line
    ret1, Frame1 = cap.read()
    cap.set(cv2.CAP_PROP_POS_MSEC, (count * 500))  # added this line
    ret2, Frame2 = cap.read()
    
    # print('Read a new frame: ', success)

    success = ret1 and ret2

    if ret1 and ret2:
        img1 = cv2.resize(Frame1, (40, 40), interpolation=cv2.INTER_AREA)
        img2 = cv2.resize(Frame2, (40, 40), interpolation=cv2.INTER_AREA)
        features.append(np.sum(np.abs(img1-img2)))
        count = count + 1

print(features)






