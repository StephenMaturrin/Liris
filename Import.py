
#importing numpy and setting a seed for the computer's
# pseudorandom number generator. This allows us to reproduce the results from our script:
import numpy as np
np.random.seed(123)  # for reproducibility

#ext, we'll import the Sequential model type from Keras. This is simply a linear stack of neural
#  network layers, and it's perfect for the type of feed-forward CNN we're building in this tutorial.
from keras.models import Sequential

#Next, let's import the "core" layers from Keras. These are the layers that are used in almost any neural network:
from keras.layers import Dense, Dropout, Activation, Flatten

#Then, we'll import the CNN layers from Keras. These are the convolutional layers that will help us efficiently train on image data:

from keras.layers import Convolution2D, MaxPooling2D

#we'll import some utilities. This will help us transform our data later:
from keras.utils import np_utils

#The Keras library conveniently includes it already
from keras.datasets import mnist

# Load pre-shuffled MNIST data into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# (60000, 28, 28)
print (X_train.shape)

