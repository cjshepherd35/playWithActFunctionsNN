import numpy as np
from tensorflow.keras.datasets import mnist
# from keras.utils import np_utils
import othernn as nn

n_hidden = 200
num_classes = 10
epochs = 1200
minibatch_size = 64
lr = 0.01
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)


conv = nn.Convolutional((minibatch_size, 1,28,28), 3,5)
reshape = nn.Reshape((minibatch_size,5,26,26),(minibatch_size,5*26*26))
# sigmoid = nn.Sigmoid()
den1 = nn.Layer_Dense(5*26*26, n_hidden)
relu1 = nn.Activation_relu()
den2 = nn.Layer_Dense(n_hidden, n_hidden)
relu2 = nn.Activation_relu()
den3 = nn.Layer_Dense(n_hidden, num_classes)
loss_activation = nn.ASLCC()
optim = nn.Optimizer_SGD(learning_rate=lr)

for i in range(epochs):
    gen = np.random.randint(0,60_000, size=minibatch_size)
    xbatch = x_train[gen]
    ybatch = y_train[gen]
    conv.forward(xbatch)
    reshape.forward(conv.output)
    # sigmoid.forward(reshape.output)
    den1.forward(reshape.output)
    relu1.forward(den1.output)
    den2.forward(relu1.output)
    relu2.forward(den2.output)
    den3.forward(relu2.output)
    loss_activation.forward(den3.output, ybatch)
    # print(loss_activation.output.shape)
    if not i%50:
        preds = np.argmax(loss_activation.output, axis=1)
        if len(ybatch.shape) == 2:
            ybatch = np.argmax(ybatch, axis=0)
        accuracy = np.mean(preds == ybatch)
        print('accuracy: ', accuracy)

    #backpropogate
    loss_activation.backward(loss_activation.output, ybatch)
    den3.backward(loss_activation.dinputs)
    relu2.backward(den3.dinputs)
    den2.backward(relu2.dinputs)
    relu1.backward(den2.dinputs)
    den1.backward(relu1.dinputs)
    # sigmoid.backward(den1.dinputs)
    reshape.backward(den1.dinputs)
    conv.backward(reshape.dinputs)

    optim.update_params(conv)
    optim.update_params(den1)
    optim.update_params(den2)
    optim.update_params(den3)