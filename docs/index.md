
# Writeup

## Summary Video

## Introduction

Trackmania 2020 is a free racing game that allows you to create custom racetracks
and race on them. For my final project, I decided to explore Convolutional Neural Networks
by training one to drive the Trackmania car. The CNN takes in images from the game
and outputs steering and speed information to control the car. By training the
model on example gameplay consisting of screenshots and car telemetry, the model is
able to replicate the driving behavior to autonomously control the car.

## Data Collection

In order to get reliable telemetry from the car for training, I utilized a scripting platform
built for Trackmania 2020 called [OpenPlanet](https://openplanet.nl/). OpenPlanet
allowed me to run scripts inside of TrackMania and access information from the
game. This not only gave me access to the speed of the car, but it also allowed
me to track when the game was actually in progress which automated the starting/stopping
of the recording process.

I created multiple large tracks of varying difficulty, some with a significant amount of
elevation changes, and some that were completely flat. Of these tracks, I subjected them
to multiple different lighting conditions to try to get the model to generalize regardless
of lighting. Here are a few examples of the tracks that I recorded on:

In order to get the data from the [OpenPlanet script](https://github.com/JVenberg/TrackmaniaSelfDriving/blob/main/Plugin_TrackManiaCustomAPI.as) to the Python [recorder.py](https://github.com/JVenberg/TrackmaniaSelfDriving/blob/main/recorder.py)
script, I used network sockets to communicate the telemetry data from the car to the recorder.
I also used d3dshot to get fast screenshots of the game view. I choose a fixed angle car view that
did not include any of the car in it to minimize any affect that the car's pose and camera angle might have
on the training.

Additionally, I implemented controller monitoring to get the steering angle inputted by me.
I also implemented various keybinds to start/stop/pause the recording to aid in the recording process.
I also implemented a datapoint cache that would cache around 10 seconds of results before saving them
to disk. This allowed me to implement keybinds to revert data when I would crash in
to walls, take bad corners, pause the game, etc. to avoid recording bad data or the pause screen.

In total I recorded ~70,000 data points over ~2 hours of driving to construct my dataset.
I then [preprocessed that data](https://github.com/JVenberg/TrackmaniaSelfDriving/blob/main/process_data.py)
by flipping the image and steering angles to ensure that there would not be any bias in one direction
which doubled the data points to ~140,000. I then split that data 85%-15% into a training and test dataset.
I also resized each image to be 64x64 to reduce the size of the model and improve runtime performance,
since I knew that I would be aiming for realtime performance. I also converted each image to greyscale
to reduce the size of the dataset, since the color of the image has no correlation to the direction of the track.

Here is an example of what the data looks like after processing:

![Data Example](imgs/data_example.png)

## Model

The model I used was based on an NVidia paper titled [End to End Learning for Self-Driving Cars](https://arxiv.org/pdf/1604.07316.pdf).
In that paper, they train a self-driving car on real world data. I thought that the model architecture would
be a good starting point for my model. Here is the neural net that NVidia described in their paper:

![NVidia Arch](imgs/nvidia_model.png)

After experimentation, I added batch normalization layers between each convolutional layer to
improve performance on my dataset. I also scaled it to work with my 64x64 image input. I also found that
a final atan operation after the last densely connected layer slightly improved performance compared to
no operation or softsign. Here is the neural net that I used:

![TrackmaniaNet Arch](imgs/model_diagram.png)

## Challenges

There are a number of challenges/obstacles that I ran into with the data collection and training of
my model.

The first was that the speed data was between [0-~300]. That decreased the performance
of the prediction. I was able to fix this by normalized the data to be between [0-1].

The second overall issue with trying to replicate my behavior is that the act of driving in a video game
naturally produces noisy and unpredictable data. For instance, there are
many data points where my inputs were not the ideal inputs and they likely differ significantly
from other similar situations. I worked off the assumption that the errors in the data would even
out, but this issue probably explains why the accuracy of the model is fairly low, and the low quality
data likely held back the performance of the model.

Lastly, even after driving over an hour, I found that there were many holes in my dataset. For example,
by only recording ideal driving, the network rarely saw how to recover from unideal situations like facing
a wall, being largely off center, or right against walls. This resulted in poor performance when the network
got in these situations. In order to remedy this, I went back and recorded a lot of data where I would pause
the recording, put the car in an unideal situation, then resume recording the correction.


## Training & Tuning

For this project, I wanted to explore automated hyperparameter tuning. Using PyTorch resources online,
I found that RayTune allowed me to automate the parameter sweep process while also parallelizing the
training using fractional GPUs and also early termination of poorly performing trials. I implemented
a tuning script that

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

Probably the largest potential improvement would be collecting more high-quality training data.
I suspect that a large reason for the occasionally unpredictable, noisy, and non-fault-tolerant
performance of the model is the lack of large amounts of data to learn from.

## Code Citations

