import numpy as np
import math
np.random.seed(0)
from scipy import signal
############################
#using multiple activation functions per layer doesn't work...
e = math.e


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, bias=True):
        self.weights = 0.1*np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1,n_neurons))
        self.bias = bias
    def forward(self, inputs):
        self.inputs = inputs
        if self.bias:
            self.output = np.dot(inputs, self.weights) + self.biases
        elif not self.bias:
            self.output = np.dot(inputs, self.weights)


    def backward(self, dvalues):
        #gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        if self.bias:
            self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        #gradient on inputs
        self.dinputs = np.dot(dvalues, self.weights.T)




class Convolutional:
    def __init__(self, input_shape, kernel_size, depth):
        num_batch, input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.numbatch = num_batch
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (num_batch, depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (num_batch, depth, kernel_size, kernel_size)
        self.weights = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.output_shape)

    def forward(self, input):
        self.input = input
        self.output = np.zeros_like(self.biases)
        for l in range(self.numbatch):
            for i in range(self.depth):
                # print(self.input[l].shape)
                # print(self.weights[l,i].shape)
                self.output[l, i] += signal.correlate2d(self.input[l], self.weights[l, i], "valid")
        self.output += self.biases

    def backward(self, output_gradient):
        self.dweights = np.zeros(self.kernels_shape)
        self.dinput = np.zeros(self.input_shape)

        for i in range(self.numbatch):
            for j in range(self.depth):
                # print(self.input[i].shape)
                # print(output_gradient[j].shape)
                self.dweights[i, j] = signal.correlate2d(self.input[i], output_gradient[i,j], "valid")
                self.dinput[i] += signal.convolve2d(output_gradient[i,j], self.weights[i, j], "full")
        self.dbiases = output_gradient




class Sigmoid:
    def forward(self, x):
        self.output =  1 / (1 + np.exp(-x))

    def backward(self,dvalues):
        s = 1 / (1 + np.exp(-np.clip(dvalues,1e-4,1e2)))
        self.dinputs =  s * (1 - s)


class Reshape:
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input):
        self.output = np.reshape(input, self.output_shape)

    def backward(self, dvalues):
        self.dinputs = np.reshape(dvalues, self.input_shape)
    


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
