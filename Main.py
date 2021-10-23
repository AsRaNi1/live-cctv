
import cv2
import os
import numpy as np
from Bounding_boxes import boxing, run
from Euclidean_dist import euclidean_distance
from Bounding_boxes import boxing

video = input("Enter video path: ")

output = input("Enter Recording output path: ")
output_path = os.path.abspath(output)

if(video == '0'):
    cap = cv2.VideoCapture(0)

else:
    path = os.path.abspath(video)                #saves path of the video file
    cap = cv2.VideoCapture(path)                 #takes path of the video file
ret = True                                       #creates a boolean 
ret, old_frame = cap.read()                      #ret is true and the first frame of video saved in old_frame

net = cv2.dnn.readNet('yolov3-tiny.weights', 'yolov3.cfg')
    
classes = []

with open('coco.txt', 'r') as f:
    classes = f.read().splitlines()

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
   
size = (frame_width, frame_height)
rec_vid = cv2.VideoWriter(output_path, 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)
if not cap.isOpened():
    raise IOError("Cannot open webcam/Cannot read file")

while ret:
    ret, frame = cap.read()          #saves the first frame of video in frame
    sime = euclidean_distance(frame, old_frame)
    indexes = []
    boxes = []
    class_ids = []
    confidences = []
    indexes, boxes, class_ids, confidences = run(frame, net, classes)
    font = cv2.FONT_HERSHEY_PLAIN

    if len(indexes) <= 0:    #if no bounding box
        continue
    elif len(indexes) > 0:  #if bounding box is presrnt

        frame = boxing(frame, indexes, boxes, class_ids, confidences, classes, font)

        
    if (sime > 92000):
         rec_vid.write(frame)

    cv2.imshow('Input', frame)   #opens the webcam in a pop-up window
    old_frame = frame            #saves the vale of the new frame in old frame to be used later in the loop
    c = cv2.waitKey(1)           #new frame comes after () ms
    if cv2.waitKey(1) & 0xFF == ord('q'): #press q on keyboard to stop the webcam
        break

cap.release()
cv2.destroyAllWindows()          #Once out of the while loop, the pop-up window closes automatically

