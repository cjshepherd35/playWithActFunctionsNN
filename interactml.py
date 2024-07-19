import numpy as np
import math
from tensorflow.keras.datasets import mnist
np.random.seed(0)
import nnfromscratch as nn
########
#makes too many parameters but is just as good as simple neural network
########
num_classes = 10 #0-9 digits
num_feats = 28*28 #784 is 28*28, image shape, if wanting direct number
n_hidden = 256
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
act1relu = nn.Activation_relu()

den21 = nn.Layer_Dense(n_hidden, n_hidden)
# den22 = nn.Layer_Dense(n_hidden*2, n_hidden*2)
den23 = nn.Layer_Dense(n_hidden, n_hidden)
act2relu = nn.Activation_relu()

den3 = nn.Layer_Dense(n_hidden*2, num_classes)
loss_activation = nn.ASLCC()
optimizer = nn.Optimizer_SGD()

#testing not fully dense layer
for i in range(400):
    #layer 1
    gen = np.random.randint(0,60_000, size=minibatch_size)
    xbatch = x_train[gen]
    ybatch = y_train[gen]
    den1.forward(xbatch)
    act1relu.forward(den1.output)

    #layer 2
    den21.forward(act1relu.output[:,:n_hidden])
    # den22.forward(act1relu.output)
    den23.forward(act1relu.output[:,n_hidden:])
    den12acts = np.concatenate((den21.output, den23.output), axis=1)
    # den12acts = den12acts + den22.output
    act2relu.forward(den12acts)
    #layer 3 and loss
    den3.forward(act2relu.output)
    loss = loss_activation.forward(den3.output, ybatch)

    # print(loss_activation.output)
    if i %50 == 0:
        # print('loss: ', loss)
        preds = np.argmax(loss_activation.output, axis=1)
        if len(ybatch.shape) == 2:
            y_train = np.argmax(y_train, axis=1)
        accuracy = np.mean(preds == ybatch)
        print('accuracy: ', accuracy)

    #backpropogate
    loss_activation.backward(loss_activation.output, ybatch)
    den3.backward(loss_activation.dinputs)
    act2relu.backward(den3.dinputs)
    den21.backward(act2relu.dinputs[:,:n_hidden])
    # den22.backward(act2relu.dinputs)
    den23.backward(act2relu.dinputs[:,n_hidden:])
    den12acts = np.concatenate((den21.dinputs, den23.dinputs), axis=1)
    # den12acts = den12acts + den22.dinputs
    act1relu.backward(den12acts)
    den1.backward(act1relu.dinputs)

    optimizer.update_params(den1)
    optimizer.update_params(den21)
    # optimizer.update_params(den22)
    optimizer.update_params(den23)
    optimizer.update_params(den3)
    if i > 150:
        lr = 0.0001

print('using test data')
den1.forward(x_test)
act1relu.forward(den1.output)
#layer 2
den21.forward(act1relu.output[:,:n_hidden])
# den22.forward(act1relu.output)
den23.forward(act1relu.output[:,n_hidden:])
den12acts = np.concatenate((den21.output, den23.output), axis=1)
# den12acts = den12acts + den22.output
act2relu.forward(den12acts)
#layer 3 and loss
den3.forward(act2relu.output)
loss = loss_activation.forward(den3.output, y_test)
preds = np.argmax(loss_activation.output, axis=1)
if len(y_test.shape) == 2:
    y_test = np.argmax(y_test, axis=1)
accuracy = np.mean(preds == y_test)
print('accuracy: ', accuracy)