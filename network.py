import numpy as np
import math
import tensorflow.keras.datasets.mnist as mnist

# TODO:
# 1. Allow for a training set to be used, i.e. multiple inputs can be fed through the network
# 2. Include labels to determine the cost
# 3. Add a cost function


# The 'Network' class, representing a neural network instance
class Network:

    # @param 'inputs': a list of inputs for the input layer, where the number of inputs correspond to
    # to the number of neurons in the input layer
    #
    # @param 'layers': a list of length n, where n is the number of layers in the network (input, hidden & output).
    # The value of the ith item in 'layers' is the number of neurons in the ith layer
    def __init__(self, inputs, layers):
        self.n_layers = len(layers)

        self.layers = np.array([None for i in range(self.n_layers)], dtype=Layer)

        self.layers[0] = Layer(layers[0], 0, layers[1], inputs.flatten())

        for i in range(1, self.n_layers - 1):
            self.layers[i] = Layer(layers[i], layers[i - 1], layers[i+1])

        self.layers[self.n_layers - 1] = Layer(layers[self.n_layers - 1], layers[self.n_layers - 2], 0)


    def feedForward(self):
        for i in range(1, self.n_layers):
            prevLayerActivations = self.layers[i - 1].get_activations()
            weights = self.layers[i - 1].get_weights()
            biases = self.layers[i].get_biases()

            newActivations = np.sum((prevLayerActivations * weights), axis=1) + biases

            self.layers[i].set_activations(newActivations)
            
    def output(self):
        output = self.layers[2]
        print(output.get_activations())






# A 'Layer' object (i.e. input, hidden or output)
class Layer:
    # A layer can represent input, hidden or output layers depending on the parameters n_in & n_out
    def __init__(self, neurons, n_in, n_out, inputs=[]):
        self.n_in = n_in
        self.n_out = n_out

        if len(inputs) == 0:
            self.neurons = np.array([Neuron(n_out, True) for i in range(neurons)])
        else:
            self.neurons = np.array([Neuron(n_out, False, inputs[i]) for i in range(neurons)])

    def get_activations(self):
        return np.array([neuron.get_activation() for neuron in self.neurons])


    def get_weights(self):
        weights = np.empty([self.n_out, len(self.neurons)])
        
        for i in range(len(self.neurons)):
            weights[:,i] = self.neurons[i].get_weights()

        return weights

    def get_biases(self):
        return np.array([neuron.get_bias() for neuron in self.neurons])

    def set_activations(self, activations):
        for i in range(len(activations)):
            self.neurons[i].set_activation(activations[i])



# The 'Neuron' class represents a single neuron in a network layer
class Neuron:
    def __init__(self, weights, bias, activation=0):
        self.activation = activation # A neuron has an activation of 0 when it is initialised
        
        # Each neuron has a list of weights associated with it, where the number of weights for each neuron is the number
        # of neurons in the next layer (so neurons in the output layer have a weight list of size 0). Weights are
        # a random value between -1 and 1 when initialised
        self.weights = np.random.uniform(-1,1,size=weights)
        
        self.bias = np.random.uniform(-1, 1) if bias else None

    def get_weights(self):
        return self.weights

    def get_activation(self):
        return self.activation

    def get_bias(self):
        return self.bias

    def set_activation(self, activation):
        self.activation = sigmoid(activation)

# Helper functions

# Sigmoid function determines an activation for each node between 0 and 1
# given the sum of the weights * inputs and the bias, x
def sigmoid(x):
	print(x)
	return 1 / (1 + math.exp(-x))


(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255
y_train = y_train / 255

layers = [784, 15, 10]

network = Network(x_train[0], layers)

network.feedForward()

network.output()