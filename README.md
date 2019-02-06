# Mushroom-Hikes

### An app that leverages deep learning to classify mushrooms in the wild.  

## Motivation
Mushroom identification is a tough challenge. There are hundreds of thousands of species, and many are very similar to each other. To tackle this challenge, I developed an app using a crowd-sourced dataset of pictures of mushrooms. You can take a picture of a mushroom in the wild, and it'll tell you what species it is. It will be a cool tool for hikers, and even classrooms to learn about nature in the real world, without having to carry around a textbook or know anything about the topic.

## Data Preparation
I documented how I obtained, cleaned, and stitched that data together in separate notebooks. Data were obtained from the following sources:  
* the [Mushroom Observer](https://mushroomobserver.org/)
* the [Danish Svampe Atlas](https://snm.ku.dk/english/news/all_news/2018/2018.5/the-fungi-classification-challenge/)  



## Model
Model is defined in two parts. We use Tensorflow-Hub to load the pretrained weights of Inception V3. The second part adds our layers. Depending what method we call, we can either:    
1) add a single dense layer with a softmax output.  
2) add a multi-head model predicting Genus and Species.   
![System Pipeline](https://github.com/pablo-martin/Mushroom-Hikes/blob/dev/static/Model_Diagram.jpg)

## Requirements  
Full requirements are shown in requirements.txt. Main dependencies include:  
* python : 3.6.8
* tensorflow : 1.12.0
* tensorflow-hub : 0.2.0
* protobuf : 3.6.1
* flask : 1.0.2  

## Pipeline
Our dataset is fed to the model using the Tensorflow Dataset [object](https://www.tensorflow.org/api_docs/python/tf/data/Dataset). There is a parameter "balanced", which when True, will feed an equal number of images from the short tail of the distribution as well as from the long tail. For each class, the long and short tail of the ditribution is determined by the LONG_TAIL_CUTOFF flag in defaults.config  

## Execution
To run the web server, run `python front_end/application.py`.  
It is recommended, however, to be run in a production server like nginx, or gunicorn.
