import numpy as np
import pandas as pd
from random import random
from util import UTIL


class MLP(object):
    def __init__(self, num_inputs=3, hidden_layers=[3, 3], num_outputs=2):

        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs

        # create a generic representation of the layers
        layers = [num_inputs] + hidden_layers + [num_outputs]

        # create random connection weights for the layers
        weights = []
        for i in range(len(layers) - 1):
            w = np.random.rand(layers[i], layers[i + 1])
            weights.append(w)
        self.weights = weights

        # save derivatives per layer
        derivatives = []
        for i in range(len(layers) - 1):
            d = np.zeros((layers[i], layers[i + 1]))
            derivatives.append(d)
        self.derivatives = derivatives

        # save activations per layer
        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations


    def forward_propagate(self, inputs):
        # the input layer activation is just the input itself
        activations = inputs

        # save the activations for backpropogation
        self.activations[0] = activations

        
        print("ROW OUTPUT")
        print(activations)
        print(len(activations))
        print("Weights")
        print(self.weights)
        print(len(self.weights))

        # iterate through the network layers
        for i, w in enumerate(self.weights):
            # calculate matrix multiplication between previous activation and weight matrix
            net_inputs = np.dot(activations, w)

            # apply sigmoid activation function
            activations = self._sigmoid(net_inputs)

            # save the activations for backpropogation
            self.activations[i + 1] = activations

        # return output layer activation
        return activations


    def back_propagate(self, error):
        # iterate backwards through the network layers
        for i in reversed(range(len(self.derivatives))):

            # get activation for previous layer
            activations = self.activations[i+1]

            # apply sigmoid derivative function
            delta = error * self._sigmoid_derivative(activations)

            # reshape delta as to have it as a 2d array
            delta_re = delta.reshape(delta.shape[0], -1).T

            # get activations for current layer
            current_activations = self.activations[i]

            # reshape activations as to have them as a 2d column matrix
            current_activations = current_activations.reshape(current_activations.shape[0],-1)

            # save derivative after applying matrix multiplication
            self.derivatives[i] = np.dot(current_activations, delta_re)

            # backpropogate the next error
            error = np.dot(delta, self.weights[i].T)


    def train(self, inputs, targets, epochs, learning_rate):
        # now enter the training loop
        for i in range(epochs):
            sum_errors = 0

            # iterate through all the training data
            for j, input in enumerate(inputs):
                target = targets[j]

                # activate the network!
                output = self.forward_propagate(input)

                error = target - output

                self.back_propagate(error)

                # now perform gradient descent on the derivatives
                # (this will update the weights
                self.gradient_descent(learning_rate)

                # keep track of the MSE for reporting later
                sum_errors += self._mse(target, output)

            # Epoch complete, report the training error
          #

        print("Training complete!")
        print("=====")


    def gradient_descent(self, learningRate=1):
        # update the weights by stepping down the gradient
        for i in range(len(self.weights)):
            weights = self.weights[i]
            derivatives = self.derivatives[i]
            weights += derivatives * learningRate


    def _sigmoid(self, x):
        y = 1.0 / (1 + np.exp(-x))
        return y


    def _sigmoid_derivative(self, x):
        return x * (1.0 - x)


    def _mse(self, target, output):
        return np.average((target - output) ** 2)

    def _find_max_value(self, output, classes):
        idx = []
        for row in output:
            max = np.max(row)
            index= row.tolist().index(max)
            idx.append(classes[index])
        return idx

if __name__ == "__main__":

    # machine_labels = ["vendor_name","model","myct","mmin","mmax","cach","chmin","chmax","prp","erp"]
    # machine_df = UTIL.import_data(UTIL,"machine.data",machine_labels)
    # machine_df = machine_df.drop(['vendor_name','model'], axis = 1)
    # new_machines_df = UTIL.min_max_normalization(UTIL,machine_df)
    # machine_training1,machine_testing1,machine_training2,machine_testing2,machine_training3,machine_testing3,machine_training4,machine_testing4,machine_training5,machine_testing5,machine_training6,machine_testing6,machine_training7,machine_testing7,machine_training8,machine_testing8,machine_training9,machine_testing9,machine_training10,machine_testing10,machine_tuning = UTIL.stratify_and_fold_regression(UTIL, machine_df)
    # #machine_list = [machine_training1,machine_testing1,machine_training2,machine_testing2,machine_training3,machine_testing3,machine_training4,machine_testing4,machine_training5,machine_testing5,machine_training6,machine_testing6,machine_training7,machine_testing7,machine_training8,machine_testing8,machine_training9,machine_testing9,machine_training10,machine_testing10,machine_tuning]
    # #machine_training1,machine_testing1,machine_training2,machine_testing2,machine_training3,machine_testing3,machine_training4,machine_testing4,machine_training5,machine_testing5,machine_training6,machine_testing6,machine_training7,machine_testing7,machine_training8,machine_testing8,machine_training9,machine_testing9,machine_training10,machine_testing10,machine_tuning = [one_hot_code(df, 'class') for df in machine_list]
    # # number of rows    
    # machine_df_size = machine_df.shape[0]
    # machine_targets = machine_df["erp"]
    # machine_training1_size = machine_training1.shape[0]
    # machine_training1_targets = machine_training1["erp"]
    # # print(machine_training1_size)
    # # print(machine_training1_targets)
    # mlp1 = MLP(8, [5], 1)
    # print(machine_training1.to_numpy())
    # print(machine_targets.to_numpy())
    # mlp1.train(machine_training1.to_numpy(), machine_training1_targets.to_numpy(), 50, 0.1)
    # # get a prediction
    # print("make a perdiction")
    # machine_testing1_np = machine_testing1.to_numpy()
    # print("nomalized test 1 dataset", machine_testing1_np)
    # output1 = mlp1.forward_propagate(machine_testing1_np)
    # print("predictions", output1)
    glass_labels = ['id-num','retractive-index','sodium','magnesium','aluminum','silicon','potasium','calcium','barium','iron','class']
    glass_df = UTIL.import_data(UTIL, "glass.data", glass_labels)
    glass_df.drop(columns=glass_df.columns[0], axis=1, inplace=True)
    print(glass_df)
    new_glass_df = UTIL.min_max_normalization(UTIL, glass_df)
    glass_training1,glass_testing1,glass_training2,glass_testing2,glass_training3,glass_testing3,glass_training4,glass_testing4,glass_training5,glass_testing5,glass_training6,glass_testing6,glass_training7,glass_testing7,glass_training8,glass_testing8,glass_training9,glass_testing9,glass_training10,glass_testing10,glass_tuning = UTIL.stratify_and_fold_classification(UTIL, new_glass_df)

    mlp1 = MLP(10, [6,6], 7)
    print("TRAINING DF")
    print(glass_training1)
    print("TRAINING numpy")
    train1 = glass_training1.to_numpy()
    print(train1)
    print(train1.shape)
    print("Traing targets")
    targets = glass_training1["class"].to_numpy()
    print(targets)
    print(targets.shape)
    test1 = glass_testing1.to_numpy()
  
    # mlp1.train(train1, targets, 500, 0.1)
    mlp1.train(train1, targets, 10, 0.1)
    output = mlp1.forward_propagate(test1)
    print("Test Predictions\n-----------------------------------")
    # target_df = pd.DataFrame(glass_testing1["class"]).T
    target_df = pd.DataFrame(glass_testing1["class"])
    # print(pd.DataFrame(glass_testing1["class"]).T)
    classes = [1, 2, 3, 4, 5, 6, 7]
    result = mlp1._find_max_value(output, classes)
    target_df['Predicted'] = result
    # target_df.loc[len(target_df)] = result
    print(result)
    print(target_df)




