##########################################################################
# Xuying Swift and Christine Johnson, 11/2022, CSCI 447 Machine Learning
# Driver class to run ten fold on MLP, GA, DE, and PSO
##########################################################################

import pandas as pd
import numpy as np
import sys
import math

from random import random

from mlp import MLP
from utils import UTILS
from genetic_alg import GA
from diff_evolution import DE
from PSO import PSO
from mlp_helper import MLP_HELPER
from particle import Particle

# function to autopopulate a list with MLP arrays and random weight arrays
def create_population(p_size, weight_list):

    population = []
    while len(population) < p_size:
        i = 0
        if len(population) < len(weight_list):
            population.append(weight_list[i])
            i += 1
        else:
            idv = []
            for item in weight_list[-1]:
                idv.append(np.random.uniform(-0.01, 0.01, item.shape))
            population.append(idv)

    return population

if __name__ == "__main__":

    #############################################################################
    ############  Classification Instances  #####################################
    #############################################################################

    #############################################
    ##################  Glass  ##################
    #############################################

    p_size = 500
    glass_classes = [1,2,3,4,5,6,7]
    print("################# Processing Classification Glass Dataset #######################\n")
    glass_mlp_0 = MLP(10, [], 7)
    glass_mlp_1 = MLP(10, [6], 7)
    glass_mlp_2 = MLP(10, [6,6], 7)
    # returns test datasets
    glass_train_output, glass_test_output = MLP_HELPER.mlp_glass_data()
    glass_weight_list_0 = MLP_HELPER.get_mlp_weights(glass_mlp_0,glass_train_output,0.01,250)
    glass_weight_list_1 = MLP_HELPER.get_mlp_weights(glass_mlp_1,glass_train_output,0.01,100)
    glass_weight_list_2 = MLP_HELPER.get_mlp_weights(glass_mlp_2,glass_train_output,0.01,100)
    # glass_weight_list = MLP_HELPER.get_mlp_weights(glass_mlp, glass_test_output,0.01,1000)

    ################ Glass MLP ##################
    # for 0 layers
    i = 1
    avg_accuracy = 0
    avg_F1 = 0
    avg_precision = 0
    avg_recall = 0
    for (test_set, weights) in zip(glass_test_output, glass_weight_list_0):
        result = UTILS.get_performance(UTILS, glass_mlp_0, weights, glass_classes, test_set)
        loss = UTILS.calculate_loss_np(UTILS, result, glass_classes)
        print(f'Glass 0 Layer MLP {i} performance: Accuracy-> {loss["Accuracy"]}, Precision-> {loss["Precision"]}, Recall-> {loss["Recall"]}, F1-Score-> {loss["F1"]}\n')
        avg_accuracy += loss["Accuracy"]
        avg_F1 += loss["F1"]
        avg_precision += loss["Precision"]
        avg_recall += loss["Recall"]
        i += 1
    print(f'Glass 0 Layer MLP Average Performance: Accuracy-> {avg_accuracy/10}, Precision-> {avg_precision/10}, Recall-> {avg_recall/10}, F1-Score-> {avg_F1/10}\n\n')

    # for 1 layer
    i = 1
    avg_accuracy = 0
    avg_F1 = 0
    avg_precision = 0
    avg_recall = 0
    for (test_set, weights) in zip(glass_test_output, glass_weight_list_1):
        result = UTILS.get_performance(UTILS, glass_mlp_1, weights, glass_classes, test_set)
        loss = UTILS.calculate_loss_np(UTILS, result, glass_classes)
        print(f'Glass 1 Layer MLP {i} performance: Accuracy-> {loss["Accuracy"]}, Precision-> {loss["Precision"]}, Recall-> {loss["Recall"]}, F1-Score-> {loss["F1"]}\n')
        avg_accuracy += loss["Accuracy"]
        avg_F1 += loss["F1"]
        avg_precision += loss["Precision"]
        avg_recall += loss["Recall"]
        i += 1
    print(f'Glass 1 Layer MLP Average Performance: Accuracy-> {avg_accuracy/10}, Precision-> {avg_precision/10}, Recall-> {avg_recall/10}, F1-Score-> {avg_F1/10}\n\n')
    
    # for 2 layers
    i = 1
    avg_accuracy = 0
    avg_F1 = 0
    avg_precision = 0
    avg_recall = 0
    for (test_set, weights) in zip(glass_test_output, glass_weight_list_2):
        result = UTILS.get_performance(UTILS, glass_mlp_2, weights, glass_classes, test_set)
        loss = UTILS.calculate_loss_np(UTILS, result, glass_classes)
        print(f'Glass 2 Layer MLP {i} performance: Accuracy-> {loss["Accuracy"]}, Precision-> {loss["Precision"]}, Recall-> {loss["Recall"]}, F1-Score-> {loss["F1"]}\n')
        avg_accuracy += loss["Accuracy"]
        avg_F1 += loss["F1"]
        avg_precision += loss["Precision"]
        avg_recall += loss["Recall"]
        i += 1
    print(f'Glass 2 Layer MLP Average Performance: Accuracy-> {avg_accuracy/10}, Precision-> {avg_precision/10}, Recall-> {avg_recall/10}, F1-Score-> {avg_F1/10}\n\n')

    # ################ Glass GA ###################
    # create glass GA population
    # ga_popu_0 = create_population(p_size,glass_weight_list_0)
    # ga_popu_1 = create_population(p_size,glass_weight_list_1)
    # ga_popu_2 = create_population(p_size,glass_weight_list_2)
    # # for 0 layers
    # i = 1
    # avg_accuracy = 0
    # avg_F1 = 0
    # avg_precision = 0
    # avg_recall = 0
    # for test_set in glass_test_output:
    #     ga = GA("class", glass_mlp_0, ga_popu_0, num_generations=200, tournament_size=2, crossover_probability=0.9, classes = glass_classes, n_size=math.ceil(p_size*(2/3)), test_values=test_set)
    #     loss = ga.run()
    #     print(f'Glass 0 Layer GA Test {i} performance: Accuracy-> {loss["Accuracy"]}, Precision-> {loss["Precision"]}, Recall-> {loss["Recall"]}, F1-Score-> {loss["F1"]}\n')
    #     avg_accuracy += loss["Accuracy"]
    #     avg_F1 += loss["F1"]
    #     avg_precision += loss["Precision"]
    #     avg_recall += loss["Recall"]
    #     i += 1
    # print(f'Glass 0 Layer GA Average Performance: Accuracy-> {avg_accuracy/10}, Precision-> {avg_precision/10}, Recall-> {avg_recall/10}, F1-Score-> {avg_F1/10}\n\n')
    
    # # for 1 layers
    # i = 1
    # avg_accuracy = 0
    # avg_F1 = 0
    # avg_precision = 0
    # avg_recall = 0
    # for test_set in glass_test_output:
    #     ga = GA("class", glass_mlp_1, ga_popu_1, num_generations=200, tournament_size=2, crossover_probability=0.9, classes = glass_classes, n_size=math.ceil(p_size*(2/3)), test_values=test_set)
    #     loss = ga.run()
    #     print(f'Glass 1 Layer GA Test {i} performance: Accuracy-> {loss["Accuracy"]}, Precision-> {loss["Precision"]}, Recall-> {loss["Recall"]}, F1-Score-> {loss["F1"]}\n')
    #     avg_accuracy += loss["Accuracy"]
    #     avg_F1 += loss["F1"]
    #     avg_precision += loss["Precision"]
    #     avg_recall += loss["Recall"]
    #     i += 1
    # print(f'Glass 1 Layer GA Average Performance: Accuracy-> {avg_accuracy/10}, Precision-> {avg_precision/10}, Recall-> {avg_recall/10}, F1-Score-> {avg_F1/10}\n\n')
    
    # # for 2 layers
    # i = 1
    # avg_accuracy = 0
    # avg_F1 = 0
    # avg_precision = 0
    # avg_recall = 0
    # for test_set in glass_test_output:
    #     ga = GA("class", glass_mlp_2, ga_popu_2, num_generations=200, tournament_size=2, crossover_probability=0.9, classes = glass_classes, n_size=math.ceil(p_size*(2/3)), test_values=test_set)
    #     loss = ga.run()
    #     print(f'Glass 0 Layer GA Test {i} performance: Accuracy-> {loss["Accuracy"]}, Precision-> {loss["Precision"]}, Recall-> {loss["Recall"]}, F1-Score-> {loss["F1"]}\n')
    #     avg_accuracy += loss["Accuracy"]
    #     avg_F1 += loss["F1"]
    #     avg_precision += loss["Precision"]
    #     avg_recall += loss["Recall"]
    #     i += 1
    # print(f'Glass 2 Layer GA Average Performance: Accuracy-> {avg_accuracy/10}, Precision-> {avg_precision/10}, Recall-> {avg_recall/10}, F1-Score-> {avg_F1/10}\n\n')

    # ############### Glass DE ####################
    # # for 0 layers
    # i = 1
    # avg_accuracy = 0
    # avg_F1 = 0
    # avg_precision = 0
    # avg_recall = 0
    # for test_set in glass_test_output:
    #     de = DE("class",glass_mlp_0,ga_popu_0,num_generations=200,classes=glass_classes, test_values=glass_test_output[5])
    #     loss = de.run()
    #     print(f'Glass 0 Layer DE Test {i} performance: Accuracy-> {loss["Accuracy"]}, Precision-> {loss["Precision"]}, Recall-> {loss["Recall"]}, F1-Score-> {loss["F1"]}\n')
    #     avg_accuracy += loss["Accuracy"]
    #     avg_F1 += loss["F1"]
    #     avg_precision += loss["Precision"]
    #     avg_recall += loss["Recall"]
    #     i += 1
    # print(f'Glass 0 Layer DE Average Performance: Accuracy-> {avg_accuracy/10}, Precision-> {avg_precision/10}, Recall-> {avg_recall/10}, F1-Score-> {avg_F1/10}\n\n')
    
    # # for 1 layers
    # i = 1
    # avg_accuracy = 0
    # avg_F1 = 0
    # avg_precision = 0
    # avg_recall = 0
    # for test_set in glass_test_output:
    #     de = DE("class",glass_mlp_1,ga_popu_1,num_generations=200,classes=glass_classes, test_values=glass_test_output[5])
    #     loss = de.run()
    #     print(f'Glass 1 Layer DE Test {i} performance: Accuracy-> {loss["Accuracy"]}, Precision-> {loss["Precision"]}, Recall-> {loss["Recall"]}, F1-Score-> {loss["F1"]}\n')
    #     avg_accuracy += loss["Accuracy"]
    #     avg_F1 += loss["F1"]
    #     avg_precision += loss["Precision"]
    #     avg_recall += loss["Recall"]
    #     i += 1
    # print(f'Glass 1 Layer DE Average Performance: Accuracy-> {avg_accuracy/10}, Precision-> {avg_precision/10}, Recall-> {avg_recall/10}, F1-Score-> {avg_F1/10}\n\n')
    
    # # for 2 layers
    # i = 1
    # avg_accuracy = 0
    # avg_F1 = 0
    # avg_precision = 0
    # avg_recall = 0
    # for test_set in glass_test_output:
    #     de = DE("class",glass_mlp_2,ga_popu_2,num_generations=200,classes=glass_classes, test_values=glass_test_output[5])
    #     loss = de.run()
    #     print(f'Glass 2 Layers DE Test {i} performance: Accuracy-> {loss["Accuracy"]}, Precision-> {loss["Precision"]}, Recall-> {loss["Recall"]}, F1-Score-> {loss["F1"]}\n')
    #     avg_accuracy += loss["Accuracy"]
    #     avg_F1 += loss["F1"]
    #     avg_precision += loss["Precision"]
    #     avg_recall += loss["Recall"]
    #     i += 1
    # print(f'Glass 2 Layers DE Average Performance: Accuracy-> {avg_accuracy/10}, Precision-> {avg_precision/10}, Recall-> {avg_recall/10}, F1-Score-> {avg_F1/10}\n\n')

    ############### GLass PSO ####################
    p_size = 20
    all_popu_0 = create_population(p_size,glass_weight_list_0)
    all_popu_1 = create_population(p_size,glass_weight_list_1)
    all_popu_2 = create_population(p_size,glass_weight_list_2)
    num_particles = 10 
    maxiter = 30

    # for 0 layers
    i = 1
    avg_accuracy = 0
    avg_F1 = 0
    avg_precision = 0
    avg_recall = 0
    shape1 = all_popu_0[0][0].shape
    for test_set in glass_test_output:
        part = Particle(all_popu_0, p_size, glass_classes, glass_mlp_0, test_set, "class", shape1)
        part.fitness()
        pso = PSO(all_popu_0, num_particles, maxiter, p_size, glass_classes, glass_mlp_0, test_set, "class", shape1)
        print(pso.err_best_g)
        print(pso.loss_best)
        print(f'Glass 0 Layers PSO Test {i} performance: Accuracy-> {loss["Accuracy"]}, Precision-> {loss["Precision"]}, Recall-> {loss["Recall"]}, F1-Score-> {loss["F1"]}\n')
        avg_accuracy += loss["Accuracy"]
        avg_F1 += loss["F1"]
        avg_precision += loss["Precision"]
        avg_recall += loss["Recall"]
        i += 1
    print(f'Glass 0 Layers PSO Average Performance: Accuracy-> {avg_accuracy/10}, Precision-> {avg_precision/10}, Recall-> {avg_recall/10}, F1-Score-> {avg_F1/10}\n\n')

    # for 1 layers
    i = 1
    avg_accuracy = 0
    avg_F1 = 0
    avg_precision = 0
    avg_recall = 0
    shape1 = all_popu_1[0][0].shape
    shape2 = all_popu_1[0][1].shape
    for test_set in glass_test_output:
        part = Particle(all_popu_1, p_size, glass_classes, glass_mlp_1, test_set, "class", shape1, shape2)
        part.fitness()
        pso = PSO(all_popu_1, num_particles, maxiter, p_size, glass_classes, glass_mlp_1, test_set, "class", shape1, shape2)
        print(pso.err_best_g)
        print(pso.loss_best)
        print(f'Glass 1 Layers PSO Test {i} performance: Accuracy-> {loss["Accuracy"]}, Precision-> {loss["Precision"]}, Recall-> {loss["Recall"]}, F1-Score-> {loss["F1"]}\n')
        avg_accuracy += loss["Accuracy"]
        avg_F1 += loss["F1"]
        avg_precision += loss["Precision"]
        avg_recall += loss["Recall"]
        i += 1
    print(f'Glass 1 Layers PSO Average Performance: Accuracy-> {avg_accuracy/10}, Precision-> {avg_precision/10}, Recall-> {avg_recall/10}, F1-Score-> {avg_F1/10}\n\n')

    # for 2 layers
    i = 1
    avg_accuracy = 0
    avg_F1 = 0
    avg_precision = 0
    avg_recall = 0
    shape1 = all_popu_2[0][0].shape
    shape2 = all_popu_2[0][1].shape
    shape3 = all_popu_2[0][2].shape
    for test_set in glass_test_output:
        part = Particle(all_popu_2, p_size, glass_classes, glass_mlp_2, test_set, "class", shape1, shape2, shape3)
        part.fitness()
        pso = PSO(all_popu_2, num_particles, maxiter, p_size, glass_classes, glass_mlp_2, test_set, "class", shape1, shape2, shape3)
        print(pso.err_best_g)
        print(pso.loss_best)
        print(f'Glass 2 Layers PSO Test {i} performance: Accuracy-> {loss["Accuracy"]}, Precision-> {loss["Precision"]}, Recall-> {loss["Recall"]}, F1-Score-> {loss["F1"]}\n')
        avg_accuracy += loss["Accuracy"]
        avg_F1 += loss["F1"]
        avg_precision += loss["Precision"]
        avg_recall += loss["Recall"]
        i += 1
    print(f'Glass 2 Layers PSO Average Performance: Accuracy-> {avg_accuracy/10}, Precision-> {avg_precision/10}, Recall-> {avg_recall/10}, F1-Score-> {avg_F1/10}\n\n')

    
    # #############################################
    # #################  Soybean  #################
    # #############################################

    # p_size = 500
    # soy_classes = [1,2,3,4]
    # print("################# Processing Classification Soybean Dataset #######################")
    # soy_mlp = MLP(22, [6], 4)
    # # returns test datasets
    # soy_train_output, soy_test_output = MLP_HELPER.mlp_soybean_data()
    # soy_weight_list = MLP_HELPER.get_mlp_weights(soy_mlp,soy_train_output,0.01,250)
    # # soy_weight_list = MLP_HELPER.get_mlp_weights(soy_mlp, soy_test_output,0.01,1000)
    # # create soy GA population
    # ga_popu = create_population(p_size,soy_weight_list)

    # ################ Soy MLP ##################
    # i = 1
    # avg_accuracy = 0
    # avg_F1 = 0
    # avg_precision = 0
    # avg_recall = 0
    # for (test_set, weights) in zip(soy_test_output, soy_weight_list):
    #     result = UTILS.get_performance(UTILS, soy_mlp, weights, soy_classes, test_set)
    #     loss = UTILS.calculate_loss_np(UTILS, result, soy_classes)
    #     print(f'Soy MLP {i} performance: Accuracy-> {loss["Accuracy"]}, Precision-> {loss["Precision"]}, Recall-> {loss["Recall"]}, F1-Score-> {loss["F1"]}\n')
    #     avg_accuracy += loss["Accuracy"]
    #     avg_F1 += loss["F1"]
    #     avg_precision += loss["Precision"]
    #     avg_recall += loss["Recall"]
    #     i += 1
    # print(f'Soy MLP Average Performance: Accuracy-> {avg_accuracy/10}, Precision-> {avg_precision/10}, Recall-> {avg_recall/10}, F1-Score-> {avg_F1/10}\n\n')

    # ################ Soy GA ###################
    # i = 1
    # avg_accuracy = 0
    # avg_F1 = 0
    # avg_precision = 0
    # avg_recall = 0
    # for test_set in soy_test_output:
    #     ga = GA("class", soy_mlp, ga_popu, num_generations=200, tournament_size=2, crossover_probability=0.9, classes = soy_classes, n_size=math.ceil(p_size*(2/3)), test_values=test_set)
    #     loss = ga.run()
    #     print(f'Soy GA Test {i} performance: Accuracy-> {loss["Accuracy"]}, Precision-> {loss["Precision"]}, Recall-> {loss["Recall"]}, F1-Score-> {loss["F1"]}\n')
    #     avg_accuracy += loss["Accuracy"]
    #     avg_F1 += loss["F1"]
    #     avg_precision += loss["Precision"]
    #     avg_recall += loss["Recall"]
    #     i += 1
    # print(f'Soy GA Average Performance: Accuracy-> {avg_accuracy/10}, Precision-> {avg_precision/10}, Recall-> {avg_recall/10}, F1-Score-> {avg_F1/10}\n\n')

    # ############### Soy DE ####################
    # i = 1
    # avg_accuracy = 0
    # avg_F1 = 0
    # avg_precision = 0
    # avg_recall = 0
    # for test_set in soy_test_output:
    #     de = DE("class",soy_mlp,ga_popu,num_generations=200,classes=soy_classes, test_values=soy_test_output[5])
    #     loss = de.run()
    #     print(f'Soy DE Test {i} performance: Accuracy-> {loss["Accuracy"]}, Precision-> {loss["Precision"]}, Recall-> {loss["Recall"]}, F1-Score-> {loss["F1"]}\n')
    #     avg_accuracy += loss["Accuracy"]
    #     avg_F1 += loss["F1"]
    #     avg_precision += loss["Precision"]
    #     avg_recall += loss["Recall"]
    #     i += 1
    # print(f'Soy DE Average Performance: Accuracy-> {avg_accuracy/10}, Precision-> {avg_precision/10}, Recall-> {avg_recall/10}, F1-Score-> {avg_F1/10}\n\n')

    # ############### Soy PSO ####################

    # #############################################
    # #################  Cancer  ##################
    # #############################################

    # p_size = 500
    # cancer_classes = [2,4]
    # print("################# Processing Classification Cancer Dataset #######################")
    # cancer_mlp = MLP(10, [6], 2)
    # # returns test datasets
    # cancer_train_output, cancer_test_output = MLP_HELPER.mlp_cancer_data()
    # cancer_weight_list = MLP_HELPER.get_mlp_weights(cancer_mlp,cancer_train_output,0.01,250)
    # # cancer_weight_list = MLP_HELPER.get_mlp_weights(cancer_mlp, cancer_test_output,0.01,1000)
    # # create cancer GA population
    # ga_popu = create_population(p_size,cancer_weight_list)

    # ################ Cancer MLP ##################
    # i = 1
    # avg_accuracy = 0
    # avg_F1 = 0
    # avg_precision = 0
    # avg_recall = 0
    # for (test_set, weights) in zip(cancer_test_output, cancer_weight_list):
    #     result = UTILS.get_performance(UTILS, cancer_mlp, weights, cancer_classes, test_set)
    #     loss = UTILS.calculate_loss_np(UTILS, result, cancer_classes)
    #     print(f'Cancer MLP {i} performance: Accuracy-> {loss["Accuracy"]}, Precision-> {loss["Precision"]}, Recall-> {loss["Recall"]}, F1-Score-> {loss["F1"]}\n')
    #     avg_accuracy += loss["Accuracy"]
    #     avg_F1 += loss["F1"]
    #     avg_precision += loss["Precision"]
    #     avg_recall += loss["Recall"]
    #     i += 1
    # print(f'Cancer MLP Average Performance: Accuracy-> {avg_accuracy/10}, Precision-> {avg_precision/10}, Recall-> {avg_recall/10}, F1-Score-> {avg_F1/10}\n\n')

    # ################ Cancer GA ###################
    # i = 1
    # avg_accuracy = 0
    # avg_F1 = 0
    # avg_precision = 0
    # avg_recall = 0
    # for test_set in cancer_test_output:
    #     ga = GA("class", cancer_mlp, ga_popu, num_generations=200, tournament_size=2, crossover_probability=0.9, classes = cancer_classes, n_size=math.ceil(p_size*(2/3)), test_values=test_set)
    #     loss = ga.run()
    #     print(f'Cancer GA Test {i} performance: Accuracy-> {loss["Accuracy"]}, Precision-> {loss["Precision"]}, Recall-> {loss["Recall"]}, F1-Score-> {loss["F1"]}\n')
    #     avg_accuracy += loss["Accuracy"]
    #     avg_F1 += loss["F1"]
    #     avg_precision += loss["Precision"]
    #     avg_recall += loss["Recall"]
    #     i += 1
    # print(f'Cancer GA Average Performance: Accuracy-> {avg_accuracy/10}, Precision-> {avg_precision/10}, Recall-> {avg_recall/10}, F1-Score-> {avg_F1/10}\n\n')

    # ############### Cancer DE ####################
    # i = 1
    # avg_accuracy = 0
    # avg_F1 = 0
    # avg_precision = 0
    # avg_recall = 0
    # for test_set in cancer_test_output:
    #     de = DE("class",cancer_mlp,ga_popu,num_generations=200,classes=cancer_classes, test_values=cancer_test_output[5])
    #     loss = de.run()
    #     print(f'Cancer DE Test {i} performance: Accuracy-> {loss["Accuracy"]}, Precision-> {loss["Precision"]}, Recall-> {loss["Recall"]}, F1-Score-> {loss["F1"]}\n')
    #     avg_accuracy += loss["Accuracy"]
    #     avg_F1 += loss["F1"]
    #     avg_precision += loss["Precision"]
    #     avg_recall += loss["Recall"]
    #     i += 1
    # print(f'Cancer DE Average Performance: Accuracy-> {avg_accuracy/10}, Precision-> {avg_precision/10}, Recall-> {avg_recall/10}, F1-Score-> {avg_F1/10}\n\n')

    # ############### Cancer PSO ####################


    # #############################################################################
    # ################  Regression Instances  #####################################
    # #############################################################################

    # #################################################
    # ##################  Machine  ####################
    # #################################################

    # p_size = 200
    # all_popu = []
    # print("################# Processing Classification Machine Dataset #######################")
    # machine_mlp = MLP(8, [10], 1)
    # machine_train_output, machine_test_output, x_max, x_min = MLP_HELPER.mlp_machine_data()
    # machine_weight_list = MLP_HELPER.get_mlp_weights(machine_mlp, machine_train_output,0.01,500)
    # # create machine GA population
    # all_popu = create_population(p_size,machine_weight_list)

    # ################ Machine MLP #################
    # i = 1
    # avg_MAE = 0
    # avg_MSE = 0
    # avg_MdAE = 0
    # avg_MAPE = 0
    # for (test_set, weights) in zip(machine_test_output, machine_weight_list):
    #     result = UTILS.get_performance(UTILS, machine_mlp, weights, classes=None, input=test_set)
    #     loss = UTILS.calculate_loss_for_regression(UTILS, result, test_set, x_max, x_min)
    #     print(f'Machine MLP {i} performance: MAE-> {loss["MAE"]}, MSE-> {loss["MSE"]}, MdAE-> {loss["MdAE"]}, MAPE-> {loss["MAPE"]}\n')
    #     avg_MAE += loss["MAE"]
    #     avg_MSE += loss["MSE"]
    #     avg_MdAE += loss["MdAE"]
    #     avg_MAPE += loss["MAPE"]
    #     i += 1
    # print(f'Machine MLP Average Performance: MAE-> {avg_MAE/10}, MSE-> {avg_MSE/10}, MdAE-> {avg_MdAE/10}, MAPE-> {avg_MAPE/10}\n\n')

    # ################ Machine GA ###################
    # i = 1
    # avg_MAE = 0
    # avg_MSE = 0
    # avg_MdAE = 0
    # avg_MAPE = 0
    # for test_set in machine_test_output:
    #     ga = GA("regress", machine_mlp, all_popu, num_generations=250, tournament_size=2, crossover_probability=0.9, n_size=math.ceil(p_size*(2/3)), test_values=machine_test_output[5], x_max=x_max, x_min=x_min)
    #     loss = ga.run()
    #     print(f'Machine GA {i} performance: MAE-> {loss["MAE"]}, MSE-> {loss["MSE"]}, MdAE-> {loss["MdAE"]}, MAPE-> {loss["MAPE"]}\n')
    #     avg_MAE += loss["MAE"]
    #     avg_MSE += loss["MSE"]
    #     avg_MdAE += loss["MdAE"]
    #     avg_MAPE += loss["MAPE"]
    #     i += 1
    # print(f'Machine GA Average Performance: MAE-> {avg_MAE/10}, MSE-> {avg_MSE/10}, MdAE-> {avg_MdAE/10}, MAPE-> {avg_MAPE/10}\n\n')

    # ################ Machine DE ####################
    # i = 1
    # avg_performance = 0
    # avg_F1 = 0
    # avg_precision = 0
    # avg_recall = 0
    # for test_set in machine_test_output:
    #     de = DE("regress",machine_mlp,all_popu,num_generations=200, test_values=machine_test_output[5], x_max=x_max, x_min=x_min)
    #     loss = de.run()
    #     print(f'Machine DE {i} performance: MAE-> {loss["MAE"]}, MSE-> {loss["MSE"]}, MdAE-> {loss["MdAE"]}, MAPE-> {loss["MAPE"]}\n')
    #     avg_MAE += loss["MAE"]
    #     avg_MSE += loss["MSE"]
    #     avg_MdAE += loss["MdAE"]
    #     avg_MAPE += loss["MAPE"]
    #     i += 1
    # print(f'Machine DE Average Performance: MAE-> {avg_MAE/10}, MSE-> {avg_MSE/10}, MdAE-> {avg_MdAE/10}, MAPE-> {avg_MAPE/10}\n\n')

    # ################ Machine PSO ####################

    
    # #############################################
    # ##################  Forestfire  #############
    # #############################################

    # p_size = 200
    # all_popu = []
    # print("################# Processing Regression Forestfire Dataset #######################")
    # forestfires_mlp = MLP(13, [6], 1)
    # forestfires_train_output, forestfires_test_output, x_max, x_min = MLP_HELPER.mlp_forestfires_data()
    # forestfires_weight_list = MLP_HELPER.get_mlp_weights(forestfires_mlp, forestfires_train_output,0.01,500)
    # # create foresfires GA population
    # all_popu = create_population(p_size,forestfires_weight_list)

    # ################ Forest Fire MLP #################
    # i = 1
    # avg_MAE = 0
    # avg_MSE = 0
    # avg_MdAE = 0
    # avg_MAPE = 0
    # for (test_set, weights) in zip(forestfires_test_output, forestfires_weight_list):
    #     result = UTILS.get_performance(UTILS, forestfires_mlp, weights, classes=None, input=test_set)
    #     loss = UTILS.calculate_loss_for_regression(UTILS, result, test_set, x_max, x_min)
    #     print(f'Forest Fire MLP {i} performance: MAE-> {loss["MAE"]}, MSE-> {loss["MSE"]}, MdAE-> {loss["MdAE"]}, MAPE-> {loss["MAPE"]}\n')
    #     avg_MAE += loss["MAE"]
    #     avg_MSE += loss["MSE"]
    #     avg_MdAE += loss["MdAE"]
    #     avg_MAPE += loss["MAPE"]
    #     i += 1
    # print(f'Forest Fire MLP Average Performance: MAE-> {avg_MAE/10}, MSE-> {avg_MSE/10}, MdAE-> {avg_MdAE/10}, MAPE-> {avg_MAPE/10}\n\n')

    # ################ Forest Fire GA ###################
    # i = 1
    # avg_MAE = 0
    # avg_MSE = 0
    # avg_MdAE = 0
    # avg_MAPE = 0
    # for test_set in forestfires_test_output:
    #     ga = GA("regress", forestfires_mlp, all_popu, num_generations=250, tournament_size=2, crossover_probability=0.9, n_size=math.ceil(p_size*(2/3)), test_values=forestfires_test_output[5], x_max=x_max, x_min=x_min)
    #     loss = ga.run()
    #     print(f'Forest Fire GA {i} performance: MAE-> {loss["MAE"]}, MSE-> {loss["MSE"]}, MdAE-> {loss["MdAE"]}, MAPE-> {loss["MAPE"]}\n')
    #     avg_MAE += loss["MAE"]
    #     avg_MSE += loss["MSE"]
    #     avg_MdAE += loss["MdAE"]
    #     avg_MAPE += loss["MAPE"]
    #     i += 1
    # print(f'Forest Fire GA Average Performance: MAE-> {avg_MAE/10}, MSE-> {avg_MSE/10}, MdAE-> {avg_MdAE/10}, MAPE-> {avg_MAPE/10}\n\n')

    # ################ Forest Fire DE ####################
    # i = 1
    # avg_performance = 0
    # avg_F1 = 0
    # avg_precision = 0
    # avg_recall = 0
    # for test_set in forestfires_test_output:
    #     de = DE("regress",forestfires_mlp,all_popu,num_generations=200, test_values=forestfires_test_output[5], x_max=x_max, x_min=x_min)
    #     loss = de.run()
    #     print(f'Forest Fires DE {i} performance: MAE-> {loss["MAE"]}, MSE-> {loss["MSE"]}, MdAE-> {loss["MdAE"]}, MAPE-> {loss["MAPE"]}\n')
    #     avg_MAE += loss["MAE"]
    #     avg_MSE += loss["MSE"]
    #     avg_MdAE += loss["MdAE"]
    #     avg_MAPE += loss["MAPE"]
    #     i += 1
    # print(f'Forest Fires DE Average Performance: MAE-> {avg_MAE/10}, MSE-> {avg_MSE/10}, MdAE-> {avg_MdAE/10}, MAPE-> {avg_MAPE/10}\n\n')

    # ################ Forest Fire PSO ####################

    
    # #############################################
    # ##################  Abalone  ################
    # #############################################

    # p_size = 200
    # all_popu = []
    # print("################# Processing Classification Abalone Dataset #######################")
    # abalone_mlp = MLP(10, [6], 1)
    # abalone_train_output, abalone_test_output, x_max, x_min = MLP_HELPER.mlp_abalone_data()
    # abalone_weight_list = MLP_HELPER.get_mlp_weights(abalone_mlp, abalone_train_output,0.01,500)
    # # create abalone GA population
    # all_popu = create_population(p_size,abalone_weight_list)

    # ################ Abalone MLP #################
    # i = 1
    # avg_MAE = 0
    # avg_MSE = 0
    # avg_MdAE = 0
    # avg_MAPE = 0
    # for (test_set, weights) in zip(abalone_test_output, abalone_weight_list):
    #     result = UTILS.get_performance(UTILS, abalone_mlp, weights, classes=None, input=test_set)
    #     loss = UTILS.calculate_loss_for_regression(UTILS, result, test_set, x_max, x_min)
    #     print(f'Abalone MLP {i} performance: MAE-> {loss["MAE"]}, MSE-> {loss["MSE"]}, MdAE-> {loss["MdAE"]}, MAPE-> {loss["MAPE"]}\n')
    #     avg_MAE += loss["MAE"]
    #     avg_MSE += loss["MSE"]
    #     avg_MdAE += loss["MdAE"]
    #     avg_MAPE += loss["MAPE"]
    #     i += 1
    # print(f'Abalone MLP Average Performance: MAE-> {avg_MAE/10}, MSE-> {avg_MSE/10}, MdAE-> {avg_MdAE/10}, MAPE-> {avg_MAPE/10}\n\n')

    # ################ Abalone GA ###################
    # i = 1
    # avg_MAE = 0
    # avg_MSE = 0
    # avg_MdAE = 0
    # avg_MAPE = 0
    # for test_set in abalone_test_output:
    #     ga = GA("regress", abalone_mlp, all_popu, num_generations=250, tournament_size=2, crossover_probability=0.9, n_size=math.ceil(p_size*(2/3)), test_values=abalone_test_output[5], x_max=x_max, x_min=x_min)
    #     loss = ga.run()
    #     print(f'Abalone GA {i} performance: MAE-> {loss["MAE"]}, MSE-> {loss["MSE"]}, MdAE-> {loss["MdAE"]}, MAPE-> {loss["MAPE"]}\n')
    #     avg_MAE += loss["MAE"]
    #     avg_MSE += loss["MSE"]
    #     avg_MdAE += loss["MdAE"]
    #     avg_MAPE += loss["MAPE"]
    #     i += 1
    # print(f'Abalone GA Average Performance: MAE-> {avg_MAE/10}, MSE-> {avg_MSE/10}, MdAE-> {avg_MdAE/10}, MAPE-> {avg_MAPE/10}\n\n')

    # ################ Abalone DE ####################
    # i = 1
    # avg_performance = 0
    # avg_F1 = 0
    # avg_precision = 0
    # avg_recall = 0
    # for test_set in abalone_test_output:
    #     de = DE("regress",abalone_mlp,all_popu,num_generations=200, test_values=abalone_test_output[5], x_max=x_max, x_min=x_min)
    #     loss = de.run()
    #     print(f'Abalone DE {i} performance: MAE-> {loss["MAE"]}, MSE-> {loss["MSE"]}, MdAE-> {loss["MdAE"]}, MAPE-> {loss["MAPE"]}\n')
    #     avg_MAE += loss["MAE"]
    #     avg_MSE += loss["MSE"]
    #     avg_MdAE += loss["MdAE"]
    #     avg_MAPE += loss["MAPE"]
    #     i += 1
    # print(f'Abalone DE Average Performance: MAE-> {avg_MAE/10}, MSE-> {avg_MSE/10}, MdAE-> {avg_MdAE/10}, MAPE-> {avg_MAPE/10}\n\n')

    # ################ Abalone PSO ####################

