import numpy as np
import math
from tensorflow.keras.datasets import mnist
np.random.seed(0)
import newnn as nn

num_classes = 10 #0-9 digits
num_feats = 28*28 #784 is 28*28, image shape, if wanting direct number
n_hidden = 2
minibatch_size = 1
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)

#flatten images to 1D 784 features
x_train, x_test = x_train.reshape([-1, num_feats]), x_test.reshape([-1, num_feats])

#normalize images from 0:255 to 0:1
x_train, x_test = x_train/ 255., x_test/255.

# lr = 0.001

den1 = nn.Layer_Dense(num_feats, n_hidden)
act1relu = nn.Activation_relu()
den2 = nn.Layer_Dense(n_hidden, num_classes)
actsoft = nn.Activation_Softmax()
adjust = nn.Adjusters()

gen = np.random.randint(0,60_000, size=minibatch_size)
xbatch = x_train[gen]
ybatch = y_train[gen]
den1.forward(xbatch)
act1relu.forward(den1.output)
den2.forward(act1relu.output)
actsoft.forward(den2.output)

#backward pass
print(den2.weights)
adjustvalue = adjust.updatefinal(ybatch)
den2.backward(adjustvalue)
print(ybatch)
print(den2.weights)