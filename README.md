# LIVE-CCTV
To detect any reasonable change in a live cctv to avoid large storage of data. Once, we notice a change, our goal would be track that object or person causing it. We would be using Computer vision concepts. Our major focus will be on Deep Learning and will try to add as many features in the process.

![image](https://user-images.githubusercontent.com/84843295/138285360-ca8120ba-ac48-4763-b830-73a1f1a6098a.png)

## TABLE OF CONTENTS

* [About the Project](#about-the-project)
  * [Tech Stack](#tech-stack)
  * [File Structure](#file-structure)
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

Aim and Description of project:
The project aims at storing only relevant data in a CCTV camera (in this case a traffic camera) to prevent storage loss and also detects the object that is causing the change in a frame and classifies them

### Tech Stack
The project uses:
* [Anaconda (Install Jupyter notebook)](https://www.anaconda.com/products/individual)
* [Python 3](https://www.python.org/download/releases/3.0/)
* [OpenCV](https://opencv.org/)
* [numpy](https://numpy.org/)

### File Structure
```
MAIN BRANCH
📦Eklavya (Live-CCTV)                        #Contains the notes for reference 
 ┣ 📜CNN (Eklavya).docx                         #Notes on Convolutional Neural Networks
 ┣ 📜Eklavya (Linear Algebra).docx              #Notes on Linear Algebra
 ┣ 📜Eklavya (Neural Networks).docx             #Notes on Basics of Neural Networks
 ┗ 📜Handwritten_Notes.pdf                      #Handwritten notes on all topics
 
DEV BRANCH
📦Old_Live-CCTV                              #Contains the previously written code that uses Contour Detection                              
 ┣ 📂Other Methods                              #Contains a python notebook for reference to other methods of similarity checking in frames
 ┃ ┗ 📜Other Methods.ipynb                         #Python notebook for other methods
 ┣ 📜ContourDetection.py                        #Python file for Contour Detection
 ┣ 📜Euclidean_dist.py                          #Python file for calculating Euclidean Distance between 2 consecutive frames of a video
 ┣ 📜Old_Main.py                                #The Main python file for execution of the code
 ┗ 📜Read.me.txt                                #A Readme for instructions
 
📦Improved_Live-CCTV                         #Contains New code for Object Classification, Uses the pre-trained Yolo Model                     
 ┣ 📜Bounding_boxes.py                          #Python file for the Bounding Box and Object Classification
 ┣ 📜coco.txt                                   #A text filr containing all the labels
 ┣ 📜Euclidean_dist.py                          #Python file for calculating Euclidean Distance between 2 consicutive frames of a video
 ┣ 📜Main.py                                    #The Main Python file for the execution fo the code
 ┣ 📜Read.me.txt                                #A Readme for instructions
 ┣ 📜test.mp4                                   #A test video provided
 ┣ 📜yolov3-tiny.weights                        #Pre-trained weights for the code
 ┗ 📜yolov3.cfg                                 #Yolov3 Configuration file
 ```


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
* [**Result Screenshot**](result.png)
* [**New Result Video**](New_Output.avi)
<div style="width:360px;max-width:100%;"><div style="height:0;padding-bottom:56.11%;position:relative;"><iframe width="360" height="202" style="position:absolute;top:0;left:0;width:100%;height:100%;" frameBorder="0" src="https://imgflip.com/embed/5risl8"></iframe></div><p><a href="https://imgflip.com/gif/5risl8"></a></p></div>

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
* Harsh Shah
* Kush Kothari
* Sahethi



<!-- CONTRIBUTORS -->
## Contributors
* [Arnav Zutshi](https://github.com/AsRaNi1)
* [Samiul Sheik](https://github.com/Codingsam164)




<!-- ACKNOWLEDGEMENTS AND REFERENCES -->
## Acknowledgements and Resources
* [SRA VJTI](http://sra.vjti.info/) Eklavya 2020
* [Neural Networks and Deep Learning](https://coursera.org/share/15a6027f8a9d5b2014b977e555a1788a)
* [CNN](https://coursera.org/share/b5ada0e8a36a2bb04ed089d54f1ab25d)
* [Computer Vision basics](https://towardsdatascience.com/computer-vision-for-beginners-part-4-64a8d9856208)
* [Contour Detection](https://learnopencv.com/contour-detection-using-opencv-python-c/)
* [More Computer Vision](https://www.pyimagesearch.com/)
