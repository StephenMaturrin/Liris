
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

from matplotlib import pyplot as plt
# Load pre-shuffled MNIST data into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# When using the Theano backend, you must explicitly declare a dimension for the depth of the input image
#reshaping


# plt.imshow(X_train[0])
# plt.show()

X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)

# (60000, 28, 28)


#The final preprocessing step for the input data is to convert our data type to float32
#  and normalize our data values to the range [0, 1]

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

#
# print (y_train.shape)
# print (X_train.shape)

# print (X_train[1])

# Convert 1-dimensional class arrays to 10-dimensional class matrices
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

# print (Y_train[:5])


# print (Y_train.shape)
# (60000, 10)
# 60000 input : 10 classes [ 0...10]

#declaring sequential model
model = Sequential()

#CNN input layer
# model.add(Convolution2D(32,(3,3), activation = "relu", input_shape=(1,28,28)))

# model.add(Convolution2D(32, 3,3, activation='relu', input_shape=(1,28,28)))

# reul vs sigmod https://www.quora.com/What-is-special-about-rectifier-neural-units-used-in-NN-learning

model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(1,28,28),  dim_ordering='th'))
#the first 3 parameters represent the number of convolution
# filters to use, the number of rows in each convolution kernel, and the number of columns
# in each convolution kernel, respectively.
#

# We can confirm this by printing the shape of the current model output:
print (model.output_shape)