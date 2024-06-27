import numpy as np
import matplotlib.pyplot as plt
import time
import pickle

def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)

def normalization(data):
    return (data/255)


class Layer:
    def __init__(self):
        self.input = None
        self.output = None
    def forward(self, input):
        pass
    def backward(self, output_gradient, learning_rate):
        pass

class Dense(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)
    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias
    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradients = np.dot(self.weights.T, output_gradient)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return input_gradients

class Activation(Layer):
    def __init__(self, activation, activation_prime):
        super().__init__()
        self.activation = activation
        self.activation_prime = activation_prime
    def forward(self, input):
        self.input = input
        return self.activation(self.input)
    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input))

class Tanh(Activation):
    def __init__(self):
        tanh = lambda x: np.tanh(x)
        tanh_prime = lambda x: 1-np.tanh(x) ** 2
        super().__init__(tanh, tanh_prime)

class Relu(Activation):
    def __init__(self):
        relu = lambda x: np.maximum(0, x)
        relu_prime = lambda x: (x > 0) * 1
        super().__init__(relu, relu_prime)

class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1/ (1+ np.exp(-x))
        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1- s)
        super().__init__(sigmoid , sigmoid_prime)

class Softmax(Layer):
    def forward(self, input):
        tmp = np.exp(input)
        self.output = tmp / np.sum(tmp)
        return self.output
    def backward(self, output_gradient, learning_rate):
        n = np.size(self.output)
        return np.dot((np.identity(n) - self.output.T) * self.output, output_gradient)

class Network:
    def __init__(self, network, inputs, outputs):
        self.network = network
        self.inputs = inputs
        self.outputs = outputs

    def predict(self, input):
        output = input
        output = np.reshape(output  , (self.inputs, 1))
        for layer in self.network:
            output = layer.forward(output)
        return output

    def train(self, loss, loss_prime, xtrain, ytrain, epochs = 100, learning_rate = 0.01, one_hot = False, verbose = False):
        for e in range(epochs):
            error = 0 
            for x,y in zip(xtrain, ytrain):

                output = self.predict(x)
                error += loss(y, output)

                grad = loss_prime(y, output.T).T
                for layer in reversed(self.network):
                    grad = layer.backward(grad, learning_rate)
            error /= len(xtrain)

            if verbose:
                print(f"epoachs:{e+1}, error={error}")

    def test(self, xtest, ytest):
        print("----------------------TESTING----------------------")
        correct = 0
        total = 0
        for x,y in zip(xtest, ytest):
            output = self.predict(x)
            if y == np.argmax(output):
               correct += 1
            total += 1
        print(f"accuracy{correct/total}")
        return correct/total

    def import_network(self, path):
        struct = []
        with open(path, "rb") as file:
            in_network = pickle.load(file)
        inputs, outputs = in_network[0][0], in_network[0][1]
        for layer in in_network:
            if isinstance(layer, Dense):
                struct.append(layer)
            if layer == "sigmoid":
                struct.append(Sigmoid())
            if layer == "softmax":
                struct.append(Softmax())
            if layer == "tanh":
                struct.append(Tanh())
            if layer == "relu":
                struct.append(Relu())
        return Network(struct, inputs, outputs)


    def export_network(self, network, path):
        struct = network.network
        out_network = [[network.inputs, network.outputs]]
        for layer in struct:
            if isinstance(layer, Dense):
                out_network.append(layer)
            if isinstance(layer, Sigmoid):
                out_network.append("sigmoid")
            if isinstance(layer, Softmax):
                out_network.append("softmax")
            if isinstance(layer, Tanh):
                out_network.append("tanh")
            if isinstance(layer, Relu):
                out_network.append("relu")
        with open(path, "wb") as file:
            pickle.dump(out_network, file)
