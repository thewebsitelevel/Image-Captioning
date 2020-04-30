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
 7. Flask
 8. TensorFlow
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

[For further information click here !!! ](https://towardsdatascience.com/introducing-deep-learning-and-neural-networks-deep-learning-for-rookies-1-bd68f9cf5883)

## Convolutional Neural Network
**Convolutional Neural Networks** are very similar to ordinary Neural Networks from the previous chapter: they are made up of neurons that have learnable weights and biases. Each neuron receives some inputs, performs a dot product and optionally follows it with a non-linearity. The whole network still expresses a single differentiable score function: from the raw image pixels on one end to class scores at the other. And they still have a loss function (e.g. SVM/Softmax) on the last (fully-connected) layer and all the tips/tricks we developed for learning regular Neural Networks still apply.

So what changes? ConvNet architectures make the explicit assumption that the inputs are images, which allows us to encode certain properties into the architecture. These then make the forward function more efficient to implement and vastly reduce the amount of parameters in the network.

[For further information click here !!!](https://cs231n.github.io/convolutional-networks)

# TensorFlow ![alt text](https://www.tensorflow.org/site-assets/images/project-logos/tensorflow-lite-logo-social.png)

The core open source library to help you develop and train ML models. Get started quickly by running Colab notebooks directly in your browser.
[For further information click here !!!](https://www.tensorflow.org/)
**Install TensorFlow2 : **

**TensorFlow is tested and supported on the following 64-bit systems:**
  * Python 3.5–3.7
  * Ubuntu 16.04 or later
  * Windows 7 or later
  * macOS 10.12.6 (Sierra) or later (no GPU support)
  * Raspbian 9.0 or later
  
**Download a package**
```
      # Requires the latest pip
      $ pip install --upgrade pip

      # Current stable release for CPU and GPU
      $ pip install tensorflow

      # Or try the preview build (unstable)
      $ pip install tf-nightly
```


# Keras
![alt text](https://s3.amazonaws.com/keras.io/img/keras-logo-2018-large-1200.png)

**You have just found Keras.**

Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano. It was developed with a focus on enabling fast experimentation. Being able to go from idea to result with the least possible delay is key to doing good research.

Use Keras if you need a deep learning library that:

  * Allows for easy and fast prototyping (through user friendliness, modularity, and extensibility).
  * Supports both convolutional networks and recurrent networks, as well as combinations of the two.
  * Runs seamlessly on CPU and GPU.

Read the documentation at[ Keras.io.](https://keras.io/)

Keras is compatible with: **Python 2.7-3.6.**

**Install Keras from PyPI (recommended):**
Note: These installation steps assume that you are on a Linux or Mac environment. If you are on Windows, you will need to remove sudo to run the commands below.
```
   $ sudo pip install keras
```

If you are using a virtualenv, you may want to avoid using sudo:

```
   $ pip install keras
```
# Numpy
**Install numpy from PyPI (recommended):**
Note: These installation steps assume that you are on a Linux or Mac environment. If you are on Windows, you will need to remove sudo to run the commands below.
```
   $sudo pip install numpy
```

If you are using a virtualenv, you may want to avoid using sudo:

```
   $pip install numpy
```

# Pandas
**Install pandas from PyPI (recommended):**
Note: These installation steps assume that you are on a Linux or Mac environment. If you are on Windows, you will need to remove sudo to run the commands below.
```
   $sudo pip install pandas
```

If you are using a virtualenv, you may want to avoid using sudo:

```
   $pip install pandas
```


**_In this the application of CNN - ResNet50 is used_**

## ResNet50
ResNet, short for Residual Networks is a classic neural network used as a backbone for many computer vision tasks. This model was the winner of ImageNet challenge in 2015. The fundamental breakthrough with ResNet was it allowed us to train extremely deep neural networks with 150+layers successfully. Prior to ResNet training very deep neural networks was difficult due to the problem of vanishing gradients.
AlexNet, the winner of ImageNet 2012 and the model that apparently kick started the focus on deep learning had only 8 convolutional layers, the VGG network had 19 and Inception or GoogleNet had 22 layers and ResNet 152 had 152 layers. In this blog we will code a ResNet-50 that is a smaller version of ResNet 152 and frequently used as a starting point for transfer learning.

![alt text](https://miro.medium.com/max/1210/1*3ND8w0xwiK3sOYLllGaQVw.png "Revolution of Depth")

### How to use ResNet50
