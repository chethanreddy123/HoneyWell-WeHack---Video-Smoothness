# importing libraries
import cv2
import numpy as np
import imquality.brisque as brisque
import PIL.Image
import matplotlib.pyplot as plt
import statistics

Quality_Score = []
   
# Create a VideoCapture object and read from input file
cap = cv2.VideoCapture('SampleVideo')
   
# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video  file")
   
count = 1

# Read until video is completed
while(cap.isOpened()):
      
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
   
    # Display the resulting frame
    cv2.imshow('Frame', frame)



    if count%50 == 0:
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

print(statistics.pstdev(Quality_Score))
   
# Closes all the frames
cv2.destroyAllWindows()