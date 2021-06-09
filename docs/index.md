
# Writeup

## Summary Video

## Introduction

Trackmania 2020 is a free racing game that allows you to create custom racetracks
and race on them. For my final project, I decided to explore Convolutional Neural Networks
by training one to drive the Trackmania car. The CNN takes in images from the game
and outputs steering and speed information to control the car.

## Data Collection

In order to get reliable speed data from the car, I utilized a scripting platform
built for Trackmania 2020 called [OpenPlanet](https://openplanet.nl/). OpenPlanet
allowed me to run scripts inside of TrackMania and access information from the
game.

## Model

The model I used was based on an NVidia paper titled [End to End Learning for Self-Driving Cars](https://arxiv.org/pdf/1604.07316.pdf).
In that paper, they train a self-driving car on real world data. I thought that the model architecture would
be a good starting point for my model. Here is the neural net that NVidia used:

![NVidia Arch](imgs/nvidia_model.png)

After experimentation, I added batch normalization layers between each convolutional layer to
improve performance on my dataset. I also scaled it to work with my 64x64 image input. Here is the
neural net that I used:

![TrackmaniaNet Arch](imgs/model_diagram.png)

#3 Challenges

## Tuning

## Results

### Trackmania 2020 Self-Driving - Day - 1st Person
<iframe width="560" height="315" src="https://www.youtube-nocookie.com/embed/J7W1Fy5_ayA" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

### Trackmania 2020 Self-Driving - Day - 3rd Person
<iframe width="560" height="315" src="https://www.youtube-nocookie.com/embed/_x_mLMAUybc" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

### Trackmania 2020 Self-Driving - Night - 1st Person
<iframe width="560" height="315" src="https://www.youtube-nocookie.com/embed/E27esEBTplI" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

### Trackmania 2020 Self-Driving - Night - 3rd Person
<iframe width="560" height="315" src="https://www.youtube-nocookie.com/embed/q7tpanRJJZ4" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Potential Improvements
