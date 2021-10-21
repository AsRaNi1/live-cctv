import cv2
import numpy as np

def Contours(bg_sub, kernel, frame):
    #applies the background subtractor
    obj_fr = bg_sub.apply(frame)
    
    #applies CLOSE MORPHOLOGY to remove black spots (removes noises)
    obj_fr = cv2.morphologyEx(obj_fr, cv2.MORPH_CLOSE, kernel)
    
    #blurring filter for lesser noise and distortion
    obj_fr = cv2.medianBlur(obj_fr, 5)
    
    #converts RGB to binary image assigns 0 to pixels less than treshold and 255 to more than threshold
    obj_fr = cv2.threshold(obj_fr,150,255,cv2.THRESH_BINARY)
    
    _, obj_fr_1 = obj_fr
    
    #detects Contours i.e Detects the change of colour in the image
    contours, hierarchy = cv2.findContours(obj_fr_1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours] #creates arrays of areas detected in different contours
    
    return contours, areas;

def Bounding_Box(frame, contours, max_index):
    cnt = contours[max_index] #Takes contours of the maximum area
    x,y,w,h = cv2.boundingRect(cnt) #Takes the center co-ordinates , height and width
    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),7) #prints out a bounding box in the image

    return frame;