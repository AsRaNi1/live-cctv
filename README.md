# LIVE-CCTV
To detect any reasonable change in a live cctv to avoid large storage of data. Once, we notice a change, our goal would be track that object or person causing it. We would be using Computer vision concepts. Our major focus will be on Deep Learning and will try to add as many features in the process.

![image](https://user-images.githubusercontent.com/84843295/138285360-ca8120ba-ac48-4763-b830-73a1f1a6098a.png)

## TABLE OF CONTENTS

* [About the Project](#about-the-project)
  * [Tech Stack](#tech-stack)
  * [File Structure](#file-structure)
* [Approach](#approach)
* [Theory](#theory)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#usage)
* [Results and Demo](#results-and-demo)
* [Future Work](#future-work)
* [Troubleshooting](#troubleshooting)
* [Mentors](#Mentors)
* [Contributors](#contributors)
* [Acknowledgements and Resources](#acknowledgements-and-resources)



## About The Project

Aim: 
The project aims at storing only relevant data in a CCTV camera (in this case a traffic camera) to prevent storage loss and also detects the object that is causing the change in a frame and classifies them.


Description: 
The project uses Yolov3 pre-trained weights and config file. It calculates the distance between 2 consecutive frames of a video and then records if the change found is relevant. It further uses the OpenCV library's DNN module to load the Yolov3 weights and then classifies the object in front of the camera, also encloses the object detected with a bounding box

### Tech Stack
The project uses:
* [Anaconda (Install Jupyter notebook)](https://www.anaconda.com/products/individual)
* [Python 3](https://www.python.org/download/releases/3.0/)
* [OpenCV](https://opencv.org/)
* [numpy](https://numpy.org/)

### File Structure
```
MAIN BRANCH
📦Notes                                    -Contains the notes for reference 
 ┣ 📜CNN (Eklavya).pdf                         -Notes on Convolutional Neural Networks
 ┣ 📜Eklavya (Linear Algebra).pdf              -Notes on Linear Algebra
 ┣ 📜Eklavya (Neural Networks).pdf             -Notes on Basics of Neural Networks
 ┗ 📜Handwritten_Notes.pdf                     -Handwritten notes on all topics
 
📦Live_CCTV_Cosine
 ┣📜Bounding_Boxes.py                           -Python file, contains function for Object Classification, Labeling them and further bounding them
 ┣📜Cos_sim.py                                  -Python file, Containing the function to calculate Cosine Similarity between 2 consecutive frames of a video
 ┣📜Main.py                                     -The Main Python file that exexutes the code
 ┣📜labels.txt                                  -Contains the labels for the objects that it detects
 ┣📜yolov3-tiny.weights                         -Pre-trained Yolov3 weights
 ┗📜yolov3.cfg                                  -Pre-trained Yolov3 Configuration file
 
 
 📦assets
 ┣ 📜LiveCCTV_Report.pdf                       -Report on the project
 ┣ 📜Output.mp4                                -Output video from test.mp4
 ┗ 📜result.png                                -Result on an image
 
 
┣📜Bounding_Boxes.py                           -Python file, contains function for Object Classification, Labeling them and further bounding them
┣📜Euclidean_dist.py                           -Python file, Containing the function to calculate Euclidean Distance between 2 consecutive frames of a video
┣📜Main.py                                     -The Main Python file that exexutes the code
┣📜Read.me.txt                                 -Contains the instructions
┣📜coco.txt                                    -Contains the labels for the objects that it detects
┣📜test.mp4                                    -Testing video
┣📜yolov3-tiny.weights                         -Pre-trained Yolov3 weights
┗📜yolov3.cfg                                  -Pre-trained Yolov3 Configuration file
 ```
 

## Approach
The approach of this project is to basically record relevant changes in a video. By checking the similarity between 2 consecutive frames of a video we can decide wether a change is relevant or no based on the magnitude of similarity.

## Theory

### Neural Networks
A neural network is a network or circuit of neurons, or in a modern sense, an artificial neural network, composed of artificial neurons or nodes.

There are different types of Neural Networks:
* Standard Neural Networks
* Convolutional Neural Networks

### Euclidean distance
It is a method to calcuate the distance between 2 vectors in a given space. In the project Euclidean distance is used to calculate distance between image vectors of 2 consecutive frames of the video.
![image](https://user-images.githubusercontent.com/84843295/138587481-d08a0e9e-289e-47b9-bdad-9e198485e1cd.png)

### Yolo
The Yolo algorithm or You Only Look Once algorithm, is an object classification algorithm that divides the image into small rids and applies object classification to each of them. The algorithm then assigns bounding boxes to the objects which further uses NMS(Non-Max Supression).

### Flowchart for code execution
![Flowchart](https://user-images.githubusercontent.com/84843295/138587595-842bf43a-01a9-4c21-91bc-0309bb0991aa.png)


## Getting Started

### Prerequisites
  Install these on the conda prompt:
  * Python 3 (conda already consists of Python)
  * OpenCV: In conda prompt type
    ```
    conda install -c conda-forge opencv
    ```
  * numpy
    ```
    conda install -c anaconda numpy
    ```


### Installation
Clone the repo
```
git clone https://github.com/AsRaNi1/live-cctv.git
```

## Usage
After cloning the repo, open the file and run the Main file in each of the folders in the Command line
Example:
```
$ Python Main.py
```
After running the main file, it will ask for the paths, so enter the paths as asked.
```
Input video file path:
Output path:
```

## Results and Demo
The project clearly identifies an object in the video frame and assigns a bounding box to it along with labels i.e classifying the object, it also records the relevant motion taking place in the video. The video is saved in the **Path** folder.  


https://user-images.githubusercontent.com/84843295/138560049-29434b6b-98aa-49ba-bfe9-dc1a5cb64e07.mp4

![result](https://user-images.githubusercontent.com/84843295/138560124-771e0b03-b55f-4eb2-b90f-5f4afc485249.png)





## Future Work
The motion tracking algorithm the current project uses is derived from Eucidean diatnce, but this can further be bettered by the use of feature vectors i.e adding a Conv-net and then linear regression to the Euclidean distance found between 2 consecutive frames of a video. Further, the project can be enhanced by making it a Realtime car tracker which will not only be able to identify defaulters while driving but will also be able to save their numberplate and then we will be able to track their vehicle using out program in various CCTV's:

* [Realtime car tracking](https://jivp-eurasipjournals.springeropen.com/articles/10.1186/s13640-018-0374-7)
* [Using the project for self driving cars](https://www.researchgate.net/publication/348558210_Project_Report_on_the_Prototype_of_an_Automated_Self_Driving_Vehicle)


<!-- TROUBLESHOOTING -->
## Troubleshooting
* Common errors while configuring the project:
  * OpenCV not installed in conda
  * numpy not installed in conda
  * Using an older Python version
  * One can change the threshold vale for movement detection, so choose a suitable one.
  * When giving the output path in command line be sure to add .avi, _Recording.avi_
  

## Mentors
A very special thanks to the mentors!!
* [Harsh Shah](https://github.com/HarshShah03325)
* [Kush Kothari](https://github.com/kkothari2001)
* [Sahethi]()



<!-- CONTRIBUTORS -->
## Contributors
* [Arnav Zutshi](https://github.com/AsRaNi1)




<!-- ACKNOWLEDGEMENTS AND REFERENCES -->
## Acknowledgements and Resources
* [SRA VJTI](http://sra.vjti.info/) Eklavya 2020
* [Neural Networks and Deep Learning](https://coursera.org/share/15a6027f8a9d5b2014b977e555a1788a)
* [CNN](https://coursera.org/share/b5ada0e8a36a2bb04ed089d54f1ab25d)
* [Computer Vision basics](https://towardsdatascience.com/computer-vision-for-beginners-part-4-64a8d9856208)
* [Contour Detection](https://learnopencv.com/contour-detection-using-opencv-python-c/)
* [More Computer Vision](https://www.pyimagesearch.com/)
* [Report for the Project](https://github.com/AsRaNi1/live-cctv/blob/master/assets/LiveCCTV_Report.pdf)
