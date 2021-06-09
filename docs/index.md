
# Summary Video

# Introduction

Trackmania 2020 is a free racing game that allows you to create custom racetracks
and race on them. For my final project, I

# Data Collection

# Model

The model I used is based on an NVidia paper titled [End to End Learning for Self-Driving Cars](https://arxiv.org/pdf/1604.07316.pdf).
In that paper, they train a self-driving car on real world data. I thought that the model architecture would
be a good starting point for my model. Here is the neural net that NVidia used:

![NVidia Arch](imgs/nvidia_model.png)

After experimentation, I added batch normalization layers between each convolutional layer to
improve performance on my dataset. I also scaled it to work with my 64x64 image input. Here is the
neural net that I used:

# Challenges

# Tuning

# Results

## Trackmania 2020 Self-Driving - Day - 1st Person
<iframe width="560" height="315" src="https://www.youtube-nocookie.com/embed/J7W1Fy5_ayA" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Trackmania 2020 Self-Driving - Day - 3rd Person
<iframe width="560" height="315" src="https://www.youtube-nocookie.com/embed/_x_mLMAUybc" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Trackmania 2020 Self-Driving - Night - 1st Person
<iframe width="560" height="315" src="https://www.youtube-nocookie.com/embed/E27esEBTplI" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Trackmania 2020 Self-Driving - Night - 3rd Person
<iframe width="560" height="315" src="https://www.youtube-nocookie.com/embed/q7tpanRJJZ4" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

# Potential Improvements
