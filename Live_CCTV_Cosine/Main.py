#main
import cv2
import os
import numpy as np
from Bounding_boxes import run
from Cos_sim import Pycos
from Bounding_boxes import boxing
from PIL import Image

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
old_frame_rgb = cv2.cvtColor(old_frame,cv2.COLOR_BGR2RGB)
old_frame_pil = Image.fromarray(old_frame_rgb)


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
    frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)

    sem = Pycos(frame_pil, old_frame_pil)

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

        
    if (sem < 0.94):
        print(sem)
        rec_vid.write(frame)

    old_frame_pil = frame_pil            #saves the vale of the new frame in old frame to be used later in the loop
    cv2.imshow('Output', frame)
    c = cv2.waitKey(100)           #new frame comes after () ms
    if cv2.waitKey(100) & 0xFF == ord('q'): #press q on keyboard to stop the webcam
        break

cap.release()
cv2.destroyAllWindows()          #Once out of the while loop, the pop-up window closes automatically

