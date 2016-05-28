# Super helpful guide: http://stackoverflow.com/questions/34673164/how-to-train-and-tune-an-artificial-multilayer-perceptron-neural-network-using-k

import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
import matplotlib.pyplot as plt

BATCH_SIZE = 128
NB_CLASSES = 10
NB_EPOCH = 20

# Create training and test sets. This part makes sense in that the mnist
# numbers are still represented as 2d matrices, which can be visualized with
# plt.imshow(X_train[0]) and plt.show() (and still look like numbers).
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Flattens the 2d matrices out into 784-dimensional vectors (28x28), per a
# method explained here: https://www.tensorflow.org/versions/master/tutorials/mnist/beginners/index.html
# Looks like a conventional preprocessing step for perceptrons that isn't used for
# things like ConvNets. Also loses some of the 2d information that might be useful.
X_train = X_train.reshape(60000, 784) # 60,000 is the conventional size of MNIST training
X_test = X_test.reshape(10000, 784) # And this is the conventional size of the test set
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Something else to keep in mind for image preprocessing: Sample randomly from the images,
# rather than using them in their entirety. This helps mitigate the effect of small translations.

# Normalize the vectors
X_train /= 255
X_test /= 255

# Convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(y_test, NB_CLASSES)

# This is where we build the neural network. This one takes the standard form of MLP, with
# three layers: An input layer, a hidden layer and an output layer.
model = Sequential()

# Input layer
# 512 in this case is the number of output dimensions, aka hidden layer neurons
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu')) # "relu" = Rectified linear unit
# Dropout is essentially regularization for neural nets.
# Good explanation: http://iamtrask.github.io/2015/07/28/dropout/
model.add(Dropout(0.2)) # Shouldn't go higher than .25 in input layer

# Hidden layer
# 512 neurons in the hidden layer. Choosing this number is difficult, but it should generally
# be somewhere between the number of input dimensions and output dimensions.
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5)) # 0.5 is common starting point for hidden layer

# Output layer
model.add(Dense(10)) # Reduce to the 10 output dimensions. This should probably equal NB_CLASSES.
# Softmax is a common activation function for the output layer of a neural net. It's basically
# the sigmoid function for the multinomial case.
model.add(Activation('softmax'))

# This sets the backpropagation model, which in this case uses RMSProp to determine the gradient
# and optimizes using categorical cross-entropy. RMSProp takes the place of something like
# stochastic gradient descent in the learning process.
rms = RMSprop()
model.compile(loss='categorical_crossentropy', optimizer=rms)

# Useful notes here: http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf

# Fit the model to the training data, using all parameters from above.
model.fit(X_train, Y_train,
          batch_size=BATCH_SIZE, nb_epoch=NB_EPOCH,
          show_accuracy=True, verbose=2,
          validation_data=(X_test, Y_test))

# Pass test data into the model for evaluation
score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])