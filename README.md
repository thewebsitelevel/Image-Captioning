# Image-Captioning
 Image Captioning is the project in which an image is provided to a machine learning model which comprises of learned weights and in result of a input image the output we get is the caption for this image.
# Technology Used:
 1. Python
 2. Deep Neural Networks
     * Convolutional Neural Network
     * LSTM
 3. Keras
 4. Numpy
 5. Pandas
 6. json
 6. Flask
# Python
The mission of the Python Software Foundation is to promote, protect, and advance the Python programming language, and to support and facilitate the growth of a diverse and international community of Python programmers.
### How to install.
Install python3.x.x version only.

[ Refer this documentation ](https://www.python.org)
# Deep Neural Networks
**Deep learning** (also known as **deep structured learning**) is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised.

Deep learning architectures such as deep neural networks, deep belief networks, recurrent neural networks and convolutional neural networks have been applied to fields including computer vision, speech recognition, natural language processing, audio recognition, social network filtering, machine translation, bioinformatics, drug design, medical image analysis, material inspection and board game programs, where they have produced results comparable to and in some cases surpassing human expert performance.

### Definition
Deep learning is a class of machine learning algorithms that uses multiple layers to progressively extract higher level features from the raw input. For example, in image processing, lower layers may identify edges, while higher layers may identify the concepts relevant to a human such as digits or letters or faces.

[For further information refer click here !!! ](https://towardsdatascience.com/introducing-deep-learning-and-neural-networks-deep-learning-for-rookies-1-bd68f9cf5883)

## Convolutional Neural Network
**Convolutional Neural Networks** are very similar to ordinary Neural Networks from the previous chapter: they are made up of neurons that have learnable weights and biases. Each neuron receives some inputs, performs a dot product and optionally follows it with a non-linearity. The whole network still expresses a single differentiable score function: from the raw image pixels on one end to class scores at the other. And they still have a loss function (e.g. SVM/Softmax) on the last (fully-connected) layer and all the tips/tricks we developed for learning regular Neural Networks still apply.

So what changes? ConvNet architectures make the explicit assumption that the inputs are images, which allows us to encode certain properties into the architecture. These then make the forward function more efficient to implement and vastly reduce the amount of parameters in the network.

[For further information refer click here !!!](https://cs231n.github.io/convolutional-networks)

# Keras
![alt text](https://s3.amazonaws.com/keras.io/img/keras-logo-2018-large-1200.png)

**_In this the application of CNN ResNet50 is used_**

## ResNet50
ResNet, short for Residual Networks is a classic neural network used as a backbone for many computer vision tasks. This model was the winner of ImageNet challenge in 2015. The fundamental breakthrough with ResNet was it allowed us to train extremely deep neural networks with 150+layers successfully. Prior to ResNet training very deep neural networks was difficult due to the problem of vanishing gradients.
AlexNet, the winner of ImageNet 2012 and the model that apparently kick started the focus on deep learning had only 8 convolutional layers, the VGG network had 19 and Inception or GoogleNet had 22 layers and ResNet 152 had 152 layers. In this blog we will code a ResNet-50 that is a smaller version of ResNet 152 and frequently used as a starting point for transfer learning.

![alt text](https://miro.medium.com/max/1210/1*3ND8w0xwiK3sOYLllGaQVw.png "Revolution of Depth")

### How to use ResNet50
