import numpy as np
import math
from tensorflow.keras.datasets import mnist
np.random.seed(0)

############################
############################
# nnfs book page 253, starting optimizer ch. 10
############################
#using multiple activation functions per layer doesn't work...
e = math.e


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1*np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1,n_neurons))
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        #gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        #gradient on inputs
        self.dinputs = np.dot(dvalues, self.weights.T)
    
class Activation_relu:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    def backward(self, dvalues):
        #since we need to modify original variable, lets make a copy first
        self.dinputs = dvalues.copy()
        #zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0

class Activation_tanh:
    def forward(self, inputs):
        self.output = np.tanh(inputs)

    def backward(self, dvalues):
        self.dinputs = 1 - np.tanh(dvalues)**2


class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, 1, keepdims=True)
    #dvalues are the loss calculated from crossentropy backward func. aka loss_categoricalCrossentropy().dinputs
    def backward(self, dvalues):
        #create uninitialized array
        self.dinputs = np.empty_like(dvalues)

        #enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            #flatten output array
            single_output = single_output.reshape(-1, 1)
            #calculate jacobian matrix of output
            jacobianM = np.diagflat(single_output) - np.dot(single_output, single_output.T)

            #calculate sample-wise gradient and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobianM, single_dvalues)


class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
    
class Loss_categoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        # samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 101e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[:, y_true] #if doesnt work, use range(len(samples)) as first argument
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    #dvalues are output of softmax forward
    def backward(self, dvalues, y_true):
        #num samples
        samples = len(dvalues)
        #number of labels in every sample, we'll use the first sameple to count them
        labels = len(dvalues[0])

        #if labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        
        #calculate gradient
        self.dinputs = -y_true / dvalues
        #normalize gradient
        self.dinputs = self.dinputs / samples
#putting together softmax and loss derivatives.
#Activation_softmax_Loss_CateforicalCrossentropy()
class ASLCC():
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_categoricalCrossentropy()

    #forward pass
    def forward(self, inputs, y_true):
        #output layers activation funtion
        self.activation.forward(inputs)
        #set the output
        self.output = self.activation.output
        #calculate and return loss value
        return self.loss.calculate(self.output, y_true)
    #dvalues are output of softmax forward.
    def backward(self, dvalues, y_true):
        #number of samples
        samples = len(dvalues)

        #if labels are one hot encoded, turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        #copy so we can safely modify
        self.dinputs = dvalues.copy()
        #calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples

class Optimizer_SGD:

    #initialize optimizer - set settings, learning rate of 1 is default.
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
    
    #update parameters
    def update_params(self, layer):
        layer.weights -= self.learning_rate * layer.dweights
        layer.biases -= self.learning_rate * layer.dbiases 


# num_classes = 10 #0-9 digits
# num_feats = 28*28 #784 is 28*28, image shape, if wanting direct number
# n_hidden = 256
# minibatch_size = 128
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)

# #flatten images to 1D 784 features
# x_train, x_test = x_train.reshape([-1, num_feats]), x_test.reshape([-1, num_feats])

# #normalize images from 0:255 to 0:1
# x_train, x_test = x_train/ 255., x_test/255.

# #use classes.....

# lr = 0.001

# den1 = Layer_Dense(num_feats, n_hidden)
# act1relu = Activation_relu()
# act1tanh = Activation_tanh()
# den2 = Layer_Dense(n_hidden, n_hidden)

# # act2 = Activation_Softmax()
# loss_activation = ASLCC()
# optimizer = Optimizer_SGD()

# #my experimentation of using the loss function to iterate with learning rate to optimize weights.
# #something wrong with loss that gets printed, it doesnt change. 
# for i in range(800):
#     # mini_batch = random.sample(x_train)   #needs work..
#     gen = np.random.randint(0,60_000, size=minibatch_size)
#     xbatch = x_train[gen]
#     ybatch = y_train[gen]
#     den1.forward(xbatch)
#     act1relu.forward(den1.output[:,:int(n_hidden/2)])
#     act1tanh.forward(den1.output[:,int(n_hidden/2):])
    
#     act1 = np.concatenate((act1relu.output, act1tanh.output), axis=1)
#     den2.forward(act1)
#     loss = loss_activation.forward(den2.output, ybatch)

#     # print(loss_activation.output)
#     if i %100 == 0:
#         # print('loss: ', loss)
#         preds = np.argmax(loss_activation.output, axis=1)
#         if len(ybatch.shape) == 2:
#             y_train = np.argmax(y_train, axis=1)
#         accuracy = np.mean(preds == ybatch)
#         print('accuracy: ', accuracy)

#     # predictions = np.argmax(loss_activation.output, axis=1)
#     # if i % 10 == 0:
#     #     print('preds: ', predictions)

#     #now we find gradients
#     loss_activation.backward(loss_activation.output, ybatch)
    
#     den2.backward(loss_activation.dinputs)
#     act1relu.backward(den2.dinputs[:,:int(n_hidden/2)])
#     act1tanh.backward(den2.dinputs[:,int(n_hidden/2):])
#     den1acts = np.concatenate((act1relu.dinputs, act1tanh.dinputs), axis=1)
#     den1.backward(den1acts)
#     optimizer.update_params(den1)
#     optimizer.update_params(den2)
#     if i > 400:
#         lr = 0.00001


# print('using test data')
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
