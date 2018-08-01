\--- title: Udacity Lane Detection Using Semantic Segmentation abstract:
| This project implements the "Semantic Segmentation" project required
in Semester 3 of the Udacity's "Self Driving Car NanoDegree Program" ---

# Context

This project implements the "Semantic Segmentation" project required in
Semester 3 of the Udacity's ["Self Driving Car NanoDegree
Program"](https://de.udacity.com/course/self-driving-car-engineer-nanodegree--nd013)

# Pre-requisites

The project requires:

1.  **Pretrained VGG Model**: Frozen VGG model, downloadble from
    [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip).
    Look at `Readme.md` in the folder `’VGGModel’`.

2.  **Kitti Road Dataset**: Required for training and testing of the
    trained model. Downloadable from
    [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/advanced_deep_learning/data_road.zip).
    Look at `Readme.md` in the folder  
    `’KittiDataSet’`.

3.  **Python Dependencies**: All python dependencies required for
    executing the project are listed in the `environment.yml`. The first
    step of `run.sh` installs all required dependencies assuming
    `[ana]conda` environment is available. See the section 4 "Running
    the Project" below.

**Note.** Automated steps are not fully tested. Watch out for failures.

# About This Project

Starting with a pre-trained VGG model, the project add two upscaling
(specifically, deconvolution) layers in order to match the size of the
network-output to the size of the input image, which is a requirement
for semantic segmentation.

The following papers are worth reading about the topic:

  - [SegNet: A Deep Convolutional Encoder-Decoder
    Architecture](https://arxiv.org/pdf/1511.00561)

  - [Fully Convolutional Networks for Semantic
    Segmentation](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)

# Running the Project

1.  Execute `run.sh`. The script will execute semantic segmentation both
    for image inputs and video inputs. Comment out steps as required.
