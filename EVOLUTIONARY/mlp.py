import numpy as np
class MLP:
    def __init__(self, num_inputs, hidden_layers, num_outputs):
        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs
        # print()
        # print("Inputs number: {}, Hidden layer: {}, Output number: {}".format(self.num_inputs, self.hidden_layers, self.num_outputs))

        # create a generic representation of the layers
        layers = [num_inputs] + hidden_layers + [num_outputs]
        # print()
        # print("Network Structure: {}".format(layers))
        # create random connection weights for the layers
        weights = []
        for i in range(len(layers) - 1):
            w = np.random.rand(layers[i], layers[i + 1])
            weights.append(w)
        self.weights = weights
        # print()
        # print("Network weights for each layer: {}".format(self.weights))

        # save activations per layer
        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations
        # print()
        # print("Network activations for each layer: {}".format(self.activations))

        # save derivatives per layer
        derivatives = []
        for i in range(len(layers) - 1):
            d = np.zeros((layers[i], layers[i + 1]))
            derivatives.append(d)
        self.derivatives = derivatives
        # print()
        # print("Network derivatives for each layer: {}".format(self.derivatives))


    def forward_feed(self, inputs):
        activations = inputs
        # save the activations
        self.activations[0] = activations
        for i, w in enumerate(self.weights):
            # the sum function
            net_inputs = np.dot(activations, w)
            activations = self.sigmoid_function(net_inputs)
            self.activations[i + 1] = activations
        # return output

        return activations


    def back_propagate(self, error):
        # iterate backwards through the network layers
        for i in reversed(range(len(self.derivatives))):
            #print("Backward propagation for layer: ", i)
            activations = self.activations[i+1]
            # sigmoid derivative
            delta = error * self.take_derivative_of_activations(activations)
            delta_re = delta.reshape(delta.shape[0], -1).T
            # get activations for current layer
            current_activations = self.activations[i]
            current_activations = current_activations.reshape(current_activations.shape[0],-1)
            self.derivatives[i] = np.dot(current_activations, delta_re)
            error = np.dot(delta, self.weights[i].T)


    def train_network(self, inputs, targets, epochs, learning_rate):
        for i in range(epochs):
            sum_errors = 0
            # iterate through all the training data
            for j, input in enumerate(inputs):
                target = targets[j]
                output = self.forward_feed(input)
                error = target - output
                self.back_propagate(error)
                # update weights 
                self.gradient_descent(learning_rate)
                # calculating mean_squared_error 
                sum_errors += self.mean_squared_error(target, output)

        # print("Finished Training !")
        # print("***********************************************")

    def gradient_descent(self, learningRate=1):
        # stepping down the gradient to update weights
        for i in range(len(self.weights)):
            weights = self.weights[i]
            derivatives = self.derivatives[i]
            derivatives = derivatives.astype(float)
            weights += derivatives * learningRate

    def sigmoid_function(self, x):
        x = x.astype(float)
        y = 1.0 / (1 + np.exp(-x))
        return y

    def take_derivative_of_activations(self, x):
        x = x.astype(float)
        return x * (1.0 - x)

    def mean_squared_error(self, target, output):
        return np.average((target - output) ** 2)

    def find_max_value(self, output, classes):
        idx = []
        for row in output:
            max = np.max(row)
            index= row.tolist().index(max)
            idx.append(classes[index])
        return idx

    def get_weights(self):
        return self.weights
