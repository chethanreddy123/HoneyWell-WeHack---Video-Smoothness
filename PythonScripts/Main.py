import pandas as pd
import matplotlib.pyplot as plt
import PIL.Image
import cv2
from matplotlib import pyplot as plt
import numpy as np
# Only needed for web grabbing images, use cv2.imread for local images
from skimage import io
import time
import numpy as np
import imquality.brisque as brisque
import statistics


SNo = []
FinalStart = []
FinalEnd = []
FinalType = []
FinalScore = []


def is_valid(image):

    # Convert image to HSV color space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Calculate histogram of saturation channel
    s = cv2.calcHist([image], [1], None, [256], [0, 256])

    # Calculate percentage of pixels with saturation >= p
    p = 0.05
    s_perc = np.sum(s[int(p * 255):-1]) / np.prod(image.shape[0:2])

    # Percentage threshold; above: valid image, below: noise
    s_thr = 0.5
    return s_perc > s_thr, abs(s_perc - s_thr)


MyListVal = []
MyListScore = []


# Create a VideoCapture object and read from input file
cap = cv2.VideoCapture('SampleVideo3.mp4')

# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video  file")

start = time.time()

# Read until video is completed
while(cap.isOpened()):

    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:

        # Display the resulting frame
        cv2.imshow('Frame', frame)

        result, score = is_valid(frame)

        MyListVal.append(result)
        MyListScore.append(score)

        cv2.putText(frame, "{}".format(result), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

end = time.time()

# When everything done, release
# the video capture object


cap.release()

# Closes all the frames
cv2.destroyAllWindows()


Split = (end - start)/len(MyListVal)

AllIndexs = []

for i in range(len(MyListVal)):
    if MyListVal[i] == False:
        AllIndexs.append(i)


Start = End = 0

for i in range(len(AllIndexs) - 1):
    # print(AllIndexs[i])
    if AllIndexs[i] == AllIndexs[i+1] - 1:
        # print(True)
        End += 1
    else:
        FinalStart.append(AllIndexs[Start]*Split)
        FinalEnd.append(AllIndexs[End]*Split)
        FinalScore.append(round(MyListScore[AllIndexs[End]], 4))
        Start = End


for i in range(len(FinalEnd)):
    FinalType.append("BlankScreen")


for i in range(len(FinalEnd)):
    SNo.append(i+1)

##########################################################
##########################################################
##########################################################


def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()
# construct the argument parse and parse the arguments


cap = cv2.VideoCapture('SampleVideo3.mp4')

# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video  file")

# Read until video is completed:

FM = []

start = time.time()
while(cap.isOpened()):

    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:

        # Display the resulting frame

        image = frame
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        fm = variance_of_laplacian(gray)
        text = "Not Blurry"

        if fm < 100:
            text = "Blurry"

        FM.append(fm)

        cv2.imshow('Frame', frame)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

# When everything done, release
# the video capture object
cap.release()

end = time.time()


# Closes all the frames
cv2.destroyAllWindows()


Split = (end - start)/len(MyListVal)

AllIndexs = []

for i in range(len(MyListVal)):
    if FM[i] < 100:
        AllIndexs.append(i)

Start = End = 0

Sub = len(FinalEnd)

for i in range(len(AllIndexs) - 1):
    # print(AllIndexs[i])
    if AllIndexs[i] == AllIndexs[i+1] - 1:
        # print(True)
        End += 1
    else:
        FinalStart.append(AllIndexs[Start]*Split)
        FinalEnd.append(AllIndexs[End]*Split)
        FinalScore.append(round(FM[End-1], 4))
        Start = End


for i in range(len(FinalEnd) - Sub):
    FinalType.append("BlurScreen")


for i in range(Sub, len(FinalEnd)):
    SNo.append(i+1)


# importing libraries

Quality_Score = []

# Create a VideoCapture object and read from input file
cap = cv2.VideoCapture('SampleVideo3.mp4')

# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video  file")

count = 1

# Read until video is completed
while(cap.isOpened()):

    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:

        # Display the resulting frame
        cv2.imshow('Frame', frame)

        if count % 50 == 0:
            Quality_Score.append(brisque.score(frame))

        count += 1

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

# When everything done, release
# the video capture object
cap.release()


# Closes all the frames
cv2.destroyAllWindows()

# Create a VideoCapture object and read from input file
count = 0
cap = cv2.VideoCapture("SampleVideo3.mp4")
success, image = cap.read()
features = []
success = True
width = cap. get(cv2. CAP_PROP_FRAME_WIDTH)
height = cap. get(cv2. CAP_PROP_FRAME_HEIGHT)


# Read until video is completed

while success:

    cap.set(cv2.CAP_PROP_POS_MSEC, (count * 1000))  # added this line
    success, image = cap.read()
    # print('Read a new frame: ', success)
    if success:
        img = cv2.resize(image, (40, 40), interpolation=cv2.INTER_AREA)
        features.append(image)
        count = count + 1
unscaled_features = np.array(features)


L = []

if len(unscaled_features) == 1:
    i = 0
    L.append(np.sum(np.abs(unscaled_features[i]))/width*height)

for i in range(len(unscaled_features)-1):

    L.append(
        (np.sum(np.abs(unscaled_features[i]-unscaled_features[i+1])))/width*height)

# When everything done, release
# the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()

print("Video Freezing Factors", statistics.pstdev(L))


# creating the videocapture object
# and reading from the input file
# Change it to 0 if reading from webcam

cap = cv2.VideoCapture('SampleVideo3.mp4')

Scores_List = []

# used to record the time when we processed last frame
prev_frame_time = 0

# used to record the time at which we processed current frame
new_frame_time = 0


start = time.time()

# Reading the video file until finished
while(cap.isOpened()):

    # Capture frame-by-frame

    ret, frame = cap.read()

    # if video finished or no Video Input
    if not ret:
        break

    # Our operations on the frame come here
    gray = frame

    # resizing the frame size according to our need
    gray = cv2.resize(gray, (500, 300))

    # font which we will be using to display FPS
    font = cv2.FONT_HERSHEY_SIMPLEX
    # time when we finish processing for this frame
    new_frame_time = time.time()

    # Calculating the fps

    # fps will be number of frame processed in given time frame
    # since their will be most of time error of 0.001 second
    # we will be subtracting it to get more accurate result

    g = (new_frame_time-prev_frame_time)

    if g == 0:
        fps = pow(10, 10)
    else:
        fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time

    # converting the fps into integer
    fps = int(fps)

    Scores_List.append(fps)
    # converting the fps to string so that we can display it on frame
    # by using putText function
    fps = str(fps)

    # putting the FPS count on the frame
    cv2.putText(gray, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)

    # displaying the frame with fps
    cv2.imshow('frame', gray)

    # press 'Q' if you want to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
# Destroy the all windows now
cv2.destroyAllWindows()

end = time.time()

New_Score_List = []

for i in range(len(Scores_List)-1):
    New_Score_List.append(abs(Scores_List[i] - Scores_List[i+1]))



New_Score_List.append(0)

Boolean_List = []
Boolean_List.append(True)

for i in New_Score_List:
    if i > 10:
        Boolean_List.append(False)
    else:
        Boolean_List.append(True)

Boolean_List.append(True)


Split = (end - start)/len(Boolean_List)

AllIndexs = []

for i in range(len(Boolean_List)):
    if Boolean_List[i] == False:
        AllIndexs.append(i)


Start = End = 0
Sub = len(FinalEnd)


for i in range(len(AllIndexs) - 1):
    # print(AllIndexs[i])
    if AllIndexs[i] == AllIndexs[i+1] - 1:
        # print(True)
        End += 1
    else:
        FinalStart.append(round(AllIndexs[Start]*Split, 3))
        FinalEnd.append(round(AllIndexs[End]*Split, 3))
        FinalScore.append(round(New_Score_List[AllIndexs[End]], 4))
        Start = End


for i in range(len(FinalEnd) - Sub):
    FinalType.append("Lagging")



D = {
    "Start" : FinalStart,
    "End" : FinalEnd,
    "Type" : FinalType,
    "Score" : FinalScore
}


import pandas as pd


data = pd.DataFrame(D)




print(data)

data.to_excel("Result3.xlsx")


print("Video Freezing Factors", statistics.pstdev(L))
print("The Final Video Quality:",statistics.pstdev(Quality_Score))




