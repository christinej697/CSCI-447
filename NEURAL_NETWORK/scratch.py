#----------------------------------------------------
# Class to implement to a Multi-Layer Neural Network
# Xuying Swift and Christine Johnson
# November 2022, CSCI 447 Machine Learning
#----------------------------------------------------

# Standard library imports
from calendar import month
import math
from random import random, uniform
import pandas as pd
import numpy as np
from typing import Tuple
from statistics import mode
from termcolor import colored
import sys

# Local application imports
from util import UTIL


class MultiLayer_NeuralNetwork(object):
    # create a instance of multilayer nueral network
    def __init__(self, num_inputs=3, num_hidden_layers_units=[3,3], num_outputs=2):

        self.num_inputs = num_inputs
        self.num_hidden_layers_units = num_hidden_layers_units
        self.num_outputs = num_outputs

        network_layers = [num_inputs] + num_hidden_layers_units + [num_outputs]

        # create and autopopulate weight arrays with random values from -0.01 to 0.01
        weight_matrix = []
        for i in range(len(network_layers) - 1):
            weight = np.random.rand(network_layers[i], network_layers[i+1])
            weight_matrix.append(weight)
        self.weight_matrix = weight_matrix

        # create array to hold weight updates from gradient descent
        update_matrix = []
        for i in range(len(network_layers) - 1):
            delta = np.zeros((network_layers[i], network_layers[i+1]))
            update_matrix.append(delta)
        self.update_matrix = update_matrix

        # create array to hold outputs from network layers
        output_matrix = []
        for i in range(len(network_layers)):
            output = np.zeros(network_layers[i])
            output_matrix.append(output)
        self.output_matrix = output_matrix

    # perform calculations for the feedforward neural net
    def feedforward_neural_network(self, row):

        output = row

        self.output_matrix[0] = output

        # print("ROW OUTPUT")
        # print(output)
        # print(len(output))
        # print(output.shape)
        # print("Weights")
        # print(self.weight_matrix)
        # print(np.array(self.weight_matrix).shape)
        # print(type(self.weight_matrix))
        for i, w in enumerate(self.weight_matrix):
            sum = np.dot(output, w)
            # get hidden layer output
            output = self.sigmoid(sum)
            self.output_matrix[i+1] = output

        # return hidden layer output
        return output

    # perform calculations for back propagation
    def back_propagation(self, error):

        for i in reversed(range(len(self.update_matrix))):

            output = self.output_matrix[i+1]
            update = error * (output * (1.0 - output))
            new_update = update.reshape(update.shape[0], -1).T
            current_outputs = self.output_matrix[i]
            current_outputs = current_outputs.reshape(current_outputs.shape[0],-1)
            self.update_matrix[i] = np.dot(current_outputs, new_update)

            # backpropogate the next error
            error = np.dot(update, self.weight_matrix[i].T)

    def training(self, rows, rt_arr, iterations, mui):
        # now enter the training loop
        for i in range(iterations):
            sum_errors = 0

            # iterate through all the training data
            for j, row in enumerate(rows):
                rt = rt_arr[j]
                # get predicted output for given row
                yt = self.feedforward_neural_network(row)
                error = rt - yt
                self.back_propagation(error)
                # apply weight updates
                self.gradient_descent(mui)
                sum_errors += self.mean_squared_error(rt, yt)

            # Epoch complete, report the training error

        print("Training complete!")
        print("=====")


    def sigmoid(self, a):
        z_h = 1.0 / (1 + np.exp(-a))
        return z_h

    def gradient_descent(self, mui=1):
        # update the weights by stepping down the gradient
        for i in range(len(self.weight_matrix)):
            weights = self.weight_matrix[i]
            updates = self.update_matrix[i]
            weights += updates * mui

    # get mean squared error for target and predicted value
    def mean_squared_error(self, rt, yt):
        return np.average((rt - yt) ** 2)

    def find_max_value(self, output, classes):
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
    # mlp1 = MultiLayer_NeuralNetwork(8, [5], 1)
    # print(machine_training1.to_numpy())
    # print(machine_targets.to_numpy())
    # mlp1.training(machine_training1.to_numpy(), machine_training1_targets.to_numpy(), 50, 0.1)
    # # get a prediction
    # print("make a perdiction")
    # machine_testing1_np = machine_testing1.to_numpy()
    # print("nomalized test 1 dataset", machine_testing1_np)
    # output1 = mlp1.feedforward_neural_network(machine_testing1_np)
    # print("predictions", output1)
    glass_labels = ['id-num','retractive-index','sodium','magnesium','aluminum','silicon','potasium','calcium','barium','iron','class']
    glass_df = UTIL.import_data(UTIL, "glass.data", glass_labels)
    glass_df.drop(columns=glass_df.columns[0], axis=1, inplace=True)
    print(glass_df)
    new_glass_df = UTIL.min_max_normalization(UTIL, glass_df)
    glass_training1,glass_testing1,glass_training2,glass_testing2,glass_training3,glass_testing3,glass_training4,glass_testing4,glass_training5,glass_testing5,glass_training6,glass_testing6,glass_training7,glass_testing7,glass_training8,glass_testing8,glass_training9,glass_testing9,glass_training10,glass_testing10,glass_tuning = UTIL.stratify_and_fold_classification(UTIL, new_glass_df)

    mlp1 = MultiLayer_NeuralNetwork(10, [6,6], 7)
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
  
    mlp1.training(train1, targets, 500, 0.1)
    output = mlp1.feedforward_neural_network(test1)
    print("Test Predictions\n-----------------------------------")
    # target_df = pd.DataFrame(glass_testing1["class"]).T
    target_df = pd.DataFrame(glass_testing1["class"])
    # print(pd.DataFrame(glass_testing1["class"]).T)
    classes = [1, 2, 3, 4, 5, 6, 7]
    result = mlp1.find_max_value(output, classes)
    target_df['Predicted'] = result
    # target_df.loc[len(target_df)] = result
    print(result)
    print(target_df)
