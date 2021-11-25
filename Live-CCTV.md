# LIVE-CCTV 
![image](https://user-images.githubusercontent.com/84843295/142942507-231103ca-a661-4cc1-9518-1e8c62c584c2.png)

***NOTE: The following project was done under professional supervision and should not be tried at home.***

***P.S: My teammate also ditched me :)***

The project aimed to detect reasonable changes in a given input video and save the frames of the relevant footage, then detect the objects present in the frames.
This blog deals with the basic setup of the project and how one can implement it on themselves.

---
## TABLE OF CONTENTS
* [Technologies Used](#TECHNOLOGIES-USED)
  * [Python](#PYTHON)
  * [Computer Vision](#COMPUTER-VISION)
    * [OpenCV](#OpenCV)
    * [NumPy](#NumPy)
  * [Deep Learning](#VERY-DEEP-LEARNING)
    * [ResNet-18](#Resnet-18)
    * [YOLO](#YOLO)
* [Approach](#APPROACH)  
  * [Getting the Input Video](#GETTING-THE-INPUT-VIDEO)
  * [Change Detection](#CHANGE-DETECTION)
    * [Feature Vector](#FEATURE-VECTOR)
    * [Cosine Similarity](#COSINE-SIMILARITY)
  * [Object Detection and Classification](#OBJECT-DETECTION-AND-CLASSIFICATION)
* [Conclusion](#CONCLUSION)
* [Future Work and Establishments](#FUTURE-WORK-AND-ESTABLISHMENTS)
* [Acknowledgement and Resources](#ACKNOWLEDGEMENT-AND-RESOURCES)
---

## TECHNOLOGIES USED
The project uses Python 3 programming language and the computer vision field to achieve the required aim. This is further implemented with deep-learning algorithms and pre-trained models to detect objects.

### PYTHON
Python 3 programming language is used Anaconda setup. Anaconda is one of the most popular python distribution platforms. The code was originally written and ran on Jupyter-Notebook, an IDE used for Python.
The installation of anaconda can be done from the following website: 
https://www.anaconda.com/products/individual
This installs the anaconda navigator from which various other applications can be installed.

### COMPUTER VISION
The project makes mostly the use of the Computer Vision field of computers. The OpenCV and NumPy libraries of Python are used to achieve the following.
#### OpenCV
OpenCV is an open-source Computer Vision and Machine Learning library. Various modules of the OpenCV library (cv2) are used in the project. With the help of OpenCV, we can capture the video and perform various operations on its frames. OpenCV can be installed by running the following command in the anaconda prompt installed earlier:
```
conda install opencv
```
#### NumPy
NumPy is an open-source library, adding support for large, multidimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays. NumPy allows us to manipulate the arrays obtained from the pixel values of the frames that are captured with the help of OpenCV library.
NumPy can be installed by running the following command in the anaconda prompt installed earlier:
```
conda install numpy
```

### (VERY) DEEP LEARNING
Deep Learning is a type of Machine Learning that tries to enact the human brain, but we are restricted with the technology for the accuracy it achieves.
The earlier version of the project did not use a Neural Network (used Contour Detection), but the later two progressions use the DNN module of the OpenCV library and the ResNet 18 model for object detection/classification and feature vector extraction respectively.
The pre-trained YOLO (You Only Look Once) weights and configuration file are used for object classification.
<p align="center">
  <img src="https://user-images.githubusercontent.com/84843295/142945221-99feabed-d701-425e-81bb-b60050c394ce.png" />
</p>

#### ResNet-18
A ResNet is a type of Neural Network in which the output of the previous layers is used along with the output gained from the further layers to reach the required conclusion. ResNet-18 belongs to a family of ResNet that uses 18 layers for obtaining the output.
The pre-trained ResNet-18 model is loaded from the PyTorch library in Python.
The PyTorch library can be installed by running the following command in the anaconda prompt installed earlier:
```
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```
<p align="center">
  <img src = https://user-images.githubusercontent.com/84843295/143282831-d2915d2f-f657-47f3-951c-1b5a4670f951.png />

</p>

#### YOLO
YOLO (You Only Look Once not You only live once 1D10T) is an algorithm that is used for object detection and classification. It uses Deep Learning to classify objects and IoU (Intersection over Union) to apply bounding boxes. The project uses pre-trained YOLO weights and config file.

---

## APPROACH

### GETTING THE INPUT VIDEO
We get our Input video in consecutive frames by the use of the OpenCV library. With inbuilt functions we take consecutive frames one by one in a while loop to get a video as an input.
```python
import cv2
cap = cv2.VideoCapture(path)
ret = True                                       #creates a boolean 
ret, old_frame = cap.read()                      #ret is true and the first frame of video saved in old_frame

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
   
size = (frame_width, frame_height)
rec_vid = cv2.VideoWriter(output_path, 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)
while ret:
    ret, frame = cap.read()          #saves the first frame of video in frame
    cv2.imshow('Input', frame)       #opens the webcam in a pop-up window
```
To compare the 2 consecutive frames later for change detection we save the 2 frames in old_frame and frame.

### CHANGE DETECTION
To detect the changes in the frames of the video the project first **extracts the feature vectors** of the 2 consecutive frames. A Feature Vector is a 128 x 1 matrix that holds unique values at each place that define the uniqueness i.e the features of an image. This is often used in facial recognition software. The ResNet-18 Model is used to extract these feature vectors and then **cosine similarity** is applied to the feature vectors to see how similar the images are. A certain threshold value is selected according to our convenience (.i.e of how sensitive we want the information to be captured). If the Value in the Cosine similarity is high then the images are very similar, if low then vice versa. The threshold value if achieved, the program starts to record the changes in the video and stores them in the memory.

#### FEATURE VECTOR
Feature vectors are used to represent numeric or symbolic attributes, called features, of an object in a mathematical, easily interpreted way. They are important for many areas of machine learning and pattern processing. Machine learning algorithms typically require a numerical representation of objects for the algorithms to do processing and statistical analysis. Feature vectors are the equivalent of vectors of explanatory variables that are used in statistical procedures such as linear regression.
An example of Feature vector would be the RGB channels. A color can be described by how much red, blue, and green there is in it. A feature vector for this would be color = [R, G, B]. One could also describe a 3-dimensional shape with a feature vector indicating its height, width, depth, etc.
In speech recognition, features can be sound lengths, noise level, noise ratios, and more.
<p align = "center">
<img src = https://user-images.githubusercontent.com/84843295/143285284-1b326689-d252-405c-ab47-defce9d64808.png />
</p>


#### COSINE SIMILARITY
As discussed earlier cosine similarity is a means of finding out of how similar 2 given vectors in a space are. The more they are similar the smaller is the angle between them and therefore the value of similarity i.e cosθ. The feature vectors of the 2 images above give us the specific unique qualities of each frame and therefore if the vectors are somewhat closer together in space, if the angle between those 2 vectors is small that means there is no significant change in the video (like leaves moving due to wind), but if the angle exceeds a certain threshold value then the change is said to be significant (a car moving on road) and thus the program starts to record those frames. The value of Cosine varies from 0 to 1. The logic behind cosine similarity is easy to understand and can be implemented in presumably most modern programming languages.
<p align = "center">
<img src = https://user-images.githubusercontent.com/84843295/143022059-6ecbab2d-f72d-4a29-9104-d9e68e2a0a26.png />
</p>
 
<p align = "center">
<img src = https://user-images.githubusercontent.com/84843295/143022580-9a2de6b6-6ed7-4262-80d1-272fae807fc0.png />
</p>

```python
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image

def Pycos(pic_one, pic_two):
    model = models.resnet18(pretrained=True)
    #Use the model object to select the desired layer
    layer = model._modules.get('avgpool')
    model.eval()

    scaler = transforms.Resize((224, 224))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    to_tensor = transforms.ToTensor()
    
    def get_vector(img):
        #2.Create a PyTorch Variable with the transformed image
        t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))
        
        #3.Create a vector of zeros that will hold our feature vector
        #The 'avgpool' layer has an output size of 512
        my_embedding = torch.zeros(512)
        
        #4.Define a function that will copy the output of a layer
        def copy_data(m, i, o):
            my_embedding.copy_(o.data.reshape(o.data.size(1)))
            
        #5.Attach that function to our selected layer
        h = layer.register_forward_hook(copy_data)
        
        #6.Run the model on our transformed image
        model(t_img)
        
        #7.Detach our copy function from the layer
        h.remove()
        
        #8.Return the feature vector
        return my_embedding
    
    pic_one_vector = get_vector(pic_one)
    pic_two_vector = get_vector(pic_two)
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    cos_sim = cos(pic_one_vector.unsqueeze(0),
                  pic_two_vector.unsqueeze(0))
    
    return cos_sim
 ```
 The above function is used to extract the feature vector and apply the cosine similarity.
 
 Although in the earlier codes the Euclidean Distance method is used to get the similarity which has somewhat the same idea.
 ```python
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
```

### OBJECT DETECTION AND CLASSIFICATION
Object detection/classification is achieved by using Deep Learning concepts i.e pre-trained Neural Network models. The project uses the Deep Neural Network (DNN) module of the OpenCV library to load the pre-trained YOLO weights and configuration file along with the labels. The Object is detected using the DNN module and then is given a Bounding Box. Multiple bounding boxes on the same object are prevented by using the IoU concept as stated earlier. 
First the configuration file, weights and the labels are loaded by the main function.
```python
net = cv2.dnn.readNet('yolov3-tiny.weights', 'yolov3.cfg')
    
classes = []

with open('coco.txt', 'r') as f:
    classes = f.read().splitlines()
```
A while loop is then run which detects and classifies objects in each frame.
Inside while loop:
```python
 indexes = []
 boxes = []
 class_ids = []
 confidences = []
 indexes, boxes, class_ids, confidences = run(frame, net, classes)
 
 frame = boxing(frame, indexes, boxes, class_ids, confidences, classes, font)
```
Here the functions run() and boxing() are used to get the measures of the bounding box and use those measures to make one.

This program gives bounding boxes and labels along with their confidences to the object in the relevant frame that is stored from the footage. Confidence here refers to the probability of the object being the one that the model detected not the one you have on your team-mate to finish his part of the project.
![Recording (online-video-cutter com)](https://user-images.githubusercontent.com/84843295/143051049-05c71818-7506-4184-b35a-baccb28f0594.gif)

---

## (CON)CLUSION
So finally after days of researching, copy-pasting and writing code the project was completed and viola here is the [GitHub repository](https://github.com/AsRaNi1/live-cctv) of the project. You can go through the entire code and clone the repo to try it yourself. The [Demo Video](https://drive.google.com/file/d/1qCQwxaOBRgo3SBvLwEO9HVgq9zVMa2rN/view?usp=sharing) for the setup.
The flowchart for the workflow is as follows:

![Flowchart](https://user-images.githubusercontent.com/84843295/138587595-842bf43a-01a9-4c21-91bc-0309bb0991aa.png)

---
## FUTURE WORK AND ESTABLISHMENTS
The project has a vast future scope and many domains will be covered along with computer vision.
- [x] Feature Vector extraction of video frames.
- [x] Relevant Frame extraction from the input video.
- [x] Object Detection and Classification
- [x] Giving bounding boxes to the Objects detected.
- [ ] Using contour detection we can bound boxes on moving objects. Comparing the coordinates ob box from YOLO and from the contour detection we can further specify and object that is causing the change in the frame i.e the object which causes the most significant change.
- [ ] The project can be used in traffic surveillance i.e at night there are not many cars passing on the road and therefore the memory of the recording device is exploited which leads to a massive loss on large scale, with this project only the relevant footage will be recorded, thus saving storage capacity.
- [ ] The project can be used in security surveillance too for the same reason.
- [ ] With more research the efficiency and accuracy if the project can be enhanced further. The code runs better on a GPU.
---
## ACKNOWLEDGEMENT AND REFERENCES
* Special thanks to [SRA VJTI](https://sravjti.in/) for providing opportunity in Eklavya 2021.
* [Report](https://github.com/AsRaNi1/live-cctv/blob/master/Assets/LiveCCTV_Report.pdf)
* [Notes](https://github.com/AsRaNi1/live-cctv/tree/master/Notes)
* [YOLO](https://www.section.io/engineering-education/introduction-to-yolo-algorithm-for-object-detection/)
* [COSINE SIMILARITY](https://towardsdatascience.com/understanding-cosine-similarity-and-its-application-fd42f585296a)
* [FEATURE VECTOR](https://medium.com/@claudio.villar/feature-vector-the-lego-brick-of-machine-learning-9f4306cdac03)
---
#### AUTHOR: [ARNAV ZUTSHI](https://github.com/AsRaNi1)

