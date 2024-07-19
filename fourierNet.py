#is completely useless#######

import numpy as np
import math
from tensorflow.keras.datasets import mnist
np.random.seed(0)
import nnfromscratch as nn

num_classes = 10 #0-9 digits
num_feats = 28*28 #784 is 28*28, image shape, if wanting direct number
n_hidden = 512
minibatch_size = 32
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)

#flatten images to 1D 784 features
x_train, x_test = x_train.reshape([-1, num_feats]), x_test.reshape([-1, num_feats])

#normalize images from 0:255 to 0:1
x_train, x_test = x_train/ 255., x_test/255.

#use classes.....

lr = 0.001

den1 = nn.Layer_Dense(num_feats, n_hidden*2)
act1fouriersin = nn.Activation_Fouriersins()
act1fouriercos = nn.Activation_Fouriercos()
den2 = nn.Layer_Dense(n_hidden*2, n_hidden*2)
act2relu = nn.Activation_relu()
den3 = nn.Layer_Dense(n_hidden*2, num_classes)
loss_activation = nn.ASLCC()
optimizer = nn.Optimizer_SGD()


for i in range(400):
    # mini_batch = random.sample(x_train)   #needs work..
    gen = np.random.randint(0,60_000, size=minibatch_size)
    xbatch = x_train[gen]
    ybatch = y_train[gen]
    den1.forward(xbatch)
    act1fouriersin.forward(den1.output)
    act1fouriercos.forward(den1.output)
    act1fourier = act1fouriercos.output + act1fouriersin.output
    den2.forward(act1fourier)
    act2relu.forward(den2.output)
    den3.forward(act2relu.output)
    loss_activation.forward(den3.output, ybatch)

    # print(loss_activation.output)
    if i %50 == 0:
        preds = np.argmax(loss_activation.output, axis=1)
        if len(ybatch.shape) == 2:
            y_train = np.argmax(y_train, axis=1)
        accuracy = np.mean(preds == ybatch)
        print('accuracy: ', accuracy)

    #backpropogate
    loss_activation.backward(loss_activation.output, ybatch)
    den3.backward(loss_activation.dinputs)
    act2relu.backward(den3.dinputs)
    den2.backward(act2relu.dinputs)
    act1fouriersin.backward(den2.dinputs)
    act1fouriercos.backward(den2.dinputs)
    backfourier = act1fouriercos.dinputs + act1fouriersin.dinputs
    den1.backward(backfourier)

    optimizer.update_params(den1)
    optimizer.update_params(den2)
    optimizer.update_params(den3)
    if i > 150:
        lr = 0.0001

print('using test data')
den1.forward(x_test)
act1fourier.forward(den1.output)
den2.forward(act1fourier.output)
act2relu.forward(den2.output)
den3.forward(act2relu.output)
loss_activation.forward(den3.output, y_test)
preds = np.argmax(loss_activation.output, axis=1)
if len(y_test.shape) == 2:
    y_test = np.argmax(y_test, axis=1)
accuracy = np.mean(preds == y_test)
print('accuracy: ', accuracy)