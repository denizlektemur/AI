import random
import numpy as np
import math

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def sigmoid_derv(x):
    return np.multiply(x, (1.0 - x))

class neuron:
    def __init__(self, amount_of_inputs, weights = None, bias = None):
        self.inputs = []
        self.weights = []
        self.output = 0
        self.error = 0

        if bias == None:
            self.bias = random.uniform(0,1)
        else:
            self.bias = bias

        if weights == None:
            for _ in range(amount_of_inputs):
                self.weights.append(random.uniform(0,1))
        else:
            self.weights = weights
    
    def calculate_output(self, input_values):
        self.inputs = input_values
        self.output = np.dot(self.inputs, self.weights)
        self.output += self.bias
        self.output = sigmoid(self.output)
        return self.output

    def calculate_weights(self, learn_rate):
        for i in range(len(self.weights)):
            self.weights[i] += learn_rate * self.error * self.inputs[i]
        self.bias += learn_rate * self.error

class neural_network:
    def __init__(self, amount_of_inputs, amount_of_outputs, learn_rate = 0.1):
        self.hidden_layer = []
        self.output_layer = []
        self.learn_rate = learn_rate

        for _ in range(amount_of_inputs):
            self.hidden_layer.append(neuron(amount_of_inputs))
        for _ in range(amount_of_outputs):
            self.output_layer.append(neuron(len(self.hidden_layer)))
    
    def check_output(self, input_values):
        hidden_layer_outputs = []
        outputs = []

        for neuron in self.hidden_layer:
            hidden_layer_outputs.append(neuron.calculate_output(input_values))

        for neuron in self.output_layer:
            outputs.append(neuron.calculate_output(hidden_layer_outputs))

        return outputs

    def update_neurons(self):
        for neuron in self.hidden_layer:
            neuron.calculate_weights(self.learn_rate)
        for neuron in self.output_layer:
            neuron.calculate_weights(self.learn_rate)

    def backpropagation(self, expected_outputs):
        for i in range(len(self.output_layer)):
            self.output_layer[i].error = (expected_outputs[i] - self.output_layer[i].output) * sigmoid_derv(self.output_layer[i].output)
        
        for i in range(len(self.hidden_layer)):
            self.hidden_layer[i].error = 0.0
            for j in range(len(self.output_layer)):
                self.hidden_layer[i].error += (self.output_layer[j].weights[i] * self.output_layer[j].error) * sigmoid_derv(self.hidden_layer[i].output)

    def train_network(self, input_data, expected_outputs, iterations = 100000):
        for _ in range(iterations):
            for dataset in range(len(input_data)):
                self.check_output(input_data[dataset])
                self.backpropagation(expected_outputs[dataset])
                self.update_neurons()              

network = neural_network(2,1)
data_set = [[1,1], [1,0], [0,1], [0,0]]
expected_data = [[0], [1], [1], [1]]

network.train_network(data_set, expected_data)
print("expected:", expected_data[0], "| output:", round(network.check_output(data_set[0])[0]))
print("expected:", expected_data[1], "| output:", round(network.check_output(data_set[1])[0]))
print("expected:", expected_data[2], "| output:", round(network.check_output(data_set[2])[0]))
print("expected:", expected_data[3], "| output:", round(network.check_output(data_set[3])[0]))


    