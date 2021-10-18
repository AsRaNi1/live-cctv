# LIVE-CCTV
To detect any reasonable change in a live cctv to avoid large storage of data. Once, we notice a change, our goal would be track that object or person causing it. We would be using Computer vision concepts. Our major focus will be on Deep Learning and will try to add as many features in the process. 

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
* [Contributors](#contributors)
* [Acknowledgements and Resources](#acknowledgements-and-resources)



## About The Project

Aim and Description of project:
The project aims at storing only relevant data in a CCTV camera (in this case a traffic camera) to prevent storage loss and also detects the object that is causing the change in a frame.

### Tech Stack
The project uses:
* [Anaconda (Install Jupyter notebook)](https://www.anaconda.com/products/individual)
* [Python 3](https://www.python.org/download/releases/3.0/)
* [OpenCV](https://opencv.org/)
* [numpy](https://numpy.org/)

### File Structure
```
New_Live-CCTV
 ┣ coco.txt
 ┣ New_Live-CCTV.ipynb
 ┣ yolov3-tiny.weights
 ┗ yolov3.cfg
 
Old_Live-CCTV
 ┣ Main code
 ┃ ┗ Live-CCTV.ipynb
 ┗ Other Methods
 ┃ ┗ Other Methods.ipynb
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
After cloning the repo, open the file and run the .ipynb in the Jupyter notebook


## Results and Demo
The project clearly identifies an object in the video frame and assigns a bounding box to it, it also records the relevant motion taking place in the video. The video is saved in the **Users** folder with the name _Recording_.  
[**Result Screenshot**](result.png)

## Future Work
The project has a vast usage for the future. This project uses cv2, but if using pretrained algorithms, object classification can be achieved. Integrating the project with darknet framework is another way to achieve object classification which can later be used to track numberplates of a car for traffic surveillance purposes. Another future development can be to integrating a speedometer and saving the images with the car number plate in the system so as to access the images of defaulters anytime.
For works related to the vast nature of the project reffer to:

* [Realtime car tracking](https://jivp-eurasipjournals.springeropen.com/articles/10.1186/s13640-018-0374-7)
* [Using the project for self driving cars](https://www.researchgate.net/publication/348558210_Project_Report_on_the_Prototype_of_an_Automated_Self_Driving_Vehicle)


<!-- TROUBLESHOOTING -->
## Troubleshooting
* Common errors while configuring the project:
  * OpenCV not installed in conda
  * numpy not installed in conda
  * Using an older Python version
  * One can change the threshold vale for movement detection, so choose a suitable one.
  * Make sure Jupyter notebook has permission to open the webcam
  * When using a custom video be sure to write it as (r"User\address\vid.amp4")


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





