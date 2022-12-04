import numpy as np

class MLP(object):
    # initialize the network with the number of inputs, number of hidden layers and output numbers
    def __init__(self, num_inputs, hidden_layers, num_outputs):
        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs

        # connect all of the layers togeter input, hidden, output
        network_layers = [num_inputs] + hidden_layers + [num_outputs]

if __name__ == "__main__":
    mlp = MLP(3, [2], 2)