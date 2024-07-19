import numpy as np
import math
np.random.seed(0)
############################
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


    def backward(self, back_indexes, sqr=1):
        self.weights[:,back_indexes] += 0.03

    
class Adjusters:
    def updatefinal(self, backvalues):
        if len(backvalues.shape) == 1:
            self.true_index = backvalues
        elif len(backvalues).shape ==2:
            self.true_index = np.argmax(backvalues, axis=1)
        return self.true_index
    
    def updateMiddle(self,  backvalues, sqr):
        pass
    
    def sqrootadjustnum(self, backvalues):
        return np.floor(backvalues**-2)
    

    
class Activation_relu:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    def backward(self, dvalues):
        #since we need to modify original variable, lets make a copy first
        self.dinputs = dvalues.copy()
        #zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0


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




#adjuster class retrieves numbers of columns that should be adjusted