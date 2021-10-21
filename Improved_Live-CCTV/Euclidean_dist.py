#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np

def euclidean_distance(image1, image2):
    img1_array = np.asarray(image1)  #converts old_frame to an array
    img2_array = np.asarray(image2)  #converts frame into an array

    #Convert image to grayscale for better results
    gray_img1_array = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY) 
    gray_img2_array = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    #Formula for Euclidean distance
    #euclidean_distance = (sigma[(x-y)^2])^0.5
    eucdis = np.linalg.norm(gray_img1_array - gray_img2_array)
    return eucdis

