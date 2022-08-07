# Stain_Detection
Deep learning based Stain Detection for Onboard Vision Systems 

# Background
Onboard camera is widely used for ADAS (Advanced Driver Assistant System) and autonomous driving systems. However, as a vehicle operates on different conditions, camera lens may get stains (such as mud, oil, etc.) on top of it. In such case, a stain/occlusion-recognition algorithm is needed to enable the wiper to clean the lens in different ways according to the type of stains . 

In this research, we are provided with collected real videos, we extract images from videos and label them. Then we learn some classification CNN (convolutional neural networks) and  implement advanced deep learning algorithm (such as DenseNet, ShuffleNet) to recognize stained images. In order to make the algorithm realizable on the driving system, we should both pay attention to the accuracy and time response.

# Dataset Buiding
Extraction
We make a screenshot automatically every 10 frames for each video.
Classification
We divide images  into 6 types according to different modes of clean after recognition.
Dataset Augmentation
For some similar images we randomly flip them up-down or left-right in order to improve generalization ability of the model.

Training set consists of 12000 images
Testing set consists of 2400 images

# Based techniques
The project choosed two models to test: Dense net and shuffle net.

## DenseNet
Generally，deeper a convolutional networks is, more accurate it could be. However, information could vanish by the time it reaches the end (or beginning) of the network.

One possible method is to create short paths from early layers to later layers.

DenseNet applies this idea. Compared with other deep CNN, it has several strengths:

1. Relatively fewer parameters for comparable accuracy.
2. It connects all matching layers directly with each other, which ensures maximum information flow between layers. 
3. It has a regularizing effect, which makes it easier to train.
4. It relieves problems of gradient vanishing and model degradation

## ShuffleNet
Besides accuracy, speed is another important consideration. Real world tasks often aim at obtaining best accuracy under a limited time and computational budget. Therefore,  lightweight architectures are motivated

Small convolution, especially 1*1 convolution has been proved to be a good way to be ‘light’, but always companied with troubling computation complexity
 
1. ShuffleNet propose 2 methods to deal with it:
2. Pointwise group convolution, which helps to reduce computation complexity.
3. Channel shuffle, which reduces the loss of information owing to the group convolution.

