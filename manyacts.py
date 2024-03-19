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

den1 = nn.Layer_Dense(num_feats, n_hidden)
act1relu = nn.Activation_relu()
act1tanh = nn.Activation_tanh()
den2 = nn.Layer_Dense(n_hidden, n_hidden)
act2tanh = nn.Activation_tanh()
act2relu = nn.Activation_relu()
den3 = nn.Layer_Dense(n_hidden, num_classes)

# act2 = Activation_Softmax()
loss_activation = nn.ASLCC()
optimizer = nn.Optimizer_SGD()

#my experimentation of using the loss function to iterate with learning rate to optimize weights.
#something wrong with loss that gets printed, it doesnt change. 
for i in range(800):
    # mini_batch = random.sample(x_train)   #needs work..
    gen = np.random.randint(0,60_000, size=minibatch_size)
    xbatch = x_train[gen]
    ybatch = y_train[gen]
    den1.forward(xbatch)
    act1relu.forward(den1.output[:,:int(n_hidden/2)])
    act1tanh.forward(den1.output[:,int(n_hidden/2):])
    act1 = np.concatenate((act1relu.output, act1tanh.output), axis=1)
    den2.forward(act1)
    # act2tanh.forward(den1.output[:,:int(n_hidden/2)])
    # act2relu.forward(den1.output[:,int(n_hidden/2):])
    # act2 = np.concatenate((act2tanh.output, act2relu.output), axis=1)
    act2relu.forward(den2.output)
    den3.forward(act2relu.output)
    loss = loss_activation.forward(den3.output, ybatch)

    # print(loss_activation.output)
    if i %100 == 0:
        # print('loss: ', loss)
        preds = np.argmax(loss_activation.output, axis=1)
        if len(ybatch.shape) == 2:
            y_train = np.argmax(y_train, axis=1)
        accuracy = np.mean(preds == ybatch)
        print('accuracy: ', accuracy)

    # predictions = np.argmax(loss_activation.output, axis=1)
    # if i % 10 == 0:
    #     print('preds: ', predictions)

    #now we find gradients
    loss_activation.backward(loss_activation.output, ybatch)
    den3.backward(loss_activation.dinputs)
    # act2tanh.backward(den3.dinputs[:,:int(n_hidden/2)])
    # act2relu.backward(den3.dinputs[:,int(n_hidden/2):])
    # den2acts = np.concatenate((act2tanh.dinputs, act2relu.dinputs), axis=1)
    act2relu.backward(den3.dinputs)
    den2.backward(act2relu.dinputs)
    act1relu.backward(den2.dinputs[:,:int(n_hidden/2)])
    act1tanh.backward(den2.dinputs[:,int(n_hidden/2):])
    den1acts = np.concatenate((act1relu.dinputs, act1tanh.dinputs), axis=1)
    den1.backward(den1acts)
    optimizer.update_params(den1)
    optimizer.update_params(den2)
    optimizer.update_params(den3)
    if i > 400:
        lr = 0.00001


print('using test data')
# den1.forward(x_test)
# act1relu.forward(den1.output[:,:int(n_hidden/2)])
# act1tanh.forward(den1.output[:,int(n_hidden/2):])
# act1 = np.concatenate((act1relu.output, act1tanh.output), axis=1)
# den2.forward(act1)
# loss_activation.forward(den2.output, y_test)
# preds = np.argmax(loss_activation.output, axis=1)
# if len(y_test.shape) == 2:
#     y_test = np.argmax(y_test, axis=1)
# accuracy = np.mean(preds == y_test)
# print('accuracy: ', accuracy)
