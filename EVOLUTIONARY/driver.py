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

# function to print results for classification mlps
def mlp_class_results(test_output, weight_list, class_mlp, classes, df_name):
    i = 1
    avg_accuracy, avg_F1, avg_precision, avg_recall = 0,0,0,0
    for (test_set, weights) in zip(test_output, weight_list):
        result = UTILS.get_performance(UTILS, class_mlp, weights, classes, test_set)
        loss = UTILS.calculate_loss_np(UTILS, result, classes)
        # print(f'{df_name} Layer MLP {i} performance: Accuracy-> {loss["Accuracy"]}, Precision-> {loss["Precision"]}, Recall-> {loss["Recall"]}, F1-Score-> {loss["F1"]}\n')
        avg_accuracy += loss["Accuracy"]
        avg_F1 += loss["F1"]
        avg_precision += loss["Precision"]
        avg_recall += loss["Recall"]
        i += 1
    print(f'{df_name} Layer MLP Average Performance: Accuracy-> {avg_accuracy/10}, Precision-> {avg_precision/10}, Recall-> {avg_recall/10}, F1-Score-> {avg_F1/10}\n\n')

# function to print results for regression mlps
def mlp_regress_results(test_output, weight_list, regress_mlp, x_min, x_max, df_name):
    i = 1
    avg_MAE, avg_MSE, avg_MdAE, avg_MAPE = 0,0,0,0
    for (test_set, weights) in zip(test_output, weight_list):
        result = UTILS.get_performance(UTILS, regress_mlp, weights, classes=None, input=test_set)
        loss = UTILS.calculate_loss_for_regression(UTILS, result, test_set, x_max, x_min)
        # print(f'{df_name} MLP Test {i} performance: MAE-> {loss["MAE"]}, MSE-> {loss["MSE"]}, MdAE-> {loss["MdAE"]}, MAPE-> {loss["MAPE"]}\n')
        avg_MAE += loss["MAE"]
        avg_MSE += loss["MSE"]
        avg_MdAE += loss["MdAE"]
        avg_MAPE += loss["MAPE"]
        i += 1
    print(f'{df_name} MLP Average Performance: MAE-> {avg_MAE/10}, MSE-> {avg_MSE/10}, MdAE-> {avg_MdAE/10}, MAPE-> {avg_MAPE/10}\n\n')

# function to print results for class ga's
def ga_class_results(test_output, class_mlp, popu, num_gens, t_size, c_prob, classes, n_size, df_name):
    i = 1
    avg_accuracy, avg_F1, avg_precision, avg_recall = 0,0,0,0
    for test_set in test_output:
        ga = GA("class", class_mlp, popu, num_generations = num_gens, tournament_size = t_size, crossover_probability=c_prob, classes=classes, n_size=n_size, test_values=test_set)
        loss = ga.run()
        # print(f'{df_name} Layer GA Test {i} performance: Accuracy-> {loss["Accuracy"]}, Precision-> {loss["Precision"]}, Recall-> {loss["Recall"]}, F1-Score-> {loss["F1"]}\n')
        avg_accuracy += loss["Accuracy"]
        avg_F1 += loss["F1"]
        avg_precision += loss["Precision"]
        avg_recall += loss["Recall"]
        i += 1
    print(f'{df_name} Layer GA Average Performance: Accuracy-> {avg_accuracy/10}, Precision-> {avg_precision/10}, Recall-> {avg_recall/10}, F1-Score-> {avg_F1/10}\n\n')

# function to print results for regression ga's
def ga_regress_results(test_output, regress_mlp, popu, num_gens, t_size, c_prob, n_size, x_max, x_min, df_name):
    i = 1
    avg_MAE, avg_MSE, avg_MdAE, avg_MAPE = 0,0,0,0
    for test_set in test_output:
        ga = GA("regress", regress_mlp, popu, num_generations=num_gens, tournament_size=t_size, crossover_probability=c_prob, n_size=n_size, test_values=test_set, x_max=x_max, x_min=x_min)
        loss = ga.run()
        # print(f'{df_name} GA Test {i} performance: MAE-> {loss["MAE"]}, MSE-> {loss["MSE"]}, MdAE-> {loss["MdAE"]}, MAPE-> {loss["MAPE"]}\n')
        avg_MAE += loss["MAE"]
        avg_MSE += loss["MSE"]
        avg_MdAE += loss["MdAE"]
        avg_MAPE += loss["MAPE"]
        i += 1
    print(f'{df_name} GA Average Performance: MAE-> {avg_MAE/10}, MSE-> {avg_MSE/10}, MdAE-> {avg_MdAE/10}, MAPE-> {avg_MAPE/10}\n\n')

# function to print results for class de's
def de_class_results(test_output, class_mlp, popu, num_generations, classes, df_name, scale_factor):
    i = 1
    avg_accuracy, avg_F1, avg_precision, avg_recall = 0,0,0,0
    for test_set in test_output:
        de = DE("class",class_mlp,popu,num_generations=num_generations,classes=classes, test_values=test_set, scale_factor=scale_factor)
        loss = de.run()
        # print(f'{df_name} Layer DE Test {i} performance: Accuracy-> {loss["Accuracy"]}, Precision-> {loss["Precision"]}, Recall-> {loss["Recall"]}, F1-Score-> {loss["F1"]}\n')
        avg_accuracy += loss["Accuracy"]
        avg_F1 += loss["F1"]
        avg_precision += loss["Precision"]
        avg_recall += loss["Recall"]
        i += 1
    print(f'{df_name} Layer DE Average Performance: Accuracy-> {avg_accuracy/10}, Precision-> {avg_precision/10}, Recall-> {avg_recall/10}, F1-Score-> {avg_F1/10}\n\n')

# function to print results for de's
def de_regress_results(test_output, regress_mlp, popu, num_generations, x_max, x_min, df_name, scale_factor):
    i = 1
    avg_MAE, avg_MSE, avg_MdAE, avg_MAPE = 0,0,0,0
    for test_set in test_output:
        de = DE("regress",regress_mlp,popu,num_generations=num_generations, test_values=test_set, x_max=x_max, x_min=x_min, scale_factor=scale_factor)
        loss = de.run()
        # print(f'{df_name} DE Test {i} performance: MAE-> {loss["MAE"]}, MSE-> {loss["MSE"]}, MdAE-> {loss["MdAE"]}, MAPE-> {loss["MAPE"]}\n')
        avg_MAE += loss["MAE"]
        avg_MSE += loss["MSE"]
        avg_MdAE += loss["MdAE"]
        avg_MAPE += loss["MAPE"]
        i += 1
    print(f'{df_name} DE Average Performance: MAE-> {avg_MAE/10}, MSE-> {avg_MSE/10}, MdAE-> {avg_MdAE/10}, MAPE-> {avg_MAPE/10}\n\n')

def pso_class_results(test_output, class_mlp, popu, classes, df_name, p_size, num_particles, maxiter, shape1, shape2 = None, shape3 = None):
    i = 1
    avg_accuracy, avg_F1, avg_precision, avg_recall = 0,0,0,0
    for test_set in test_output:
        part = Particle(popu, p_size, classes, class_mlp, test_set, "class", shape1, shape2, shape3)
        part.fitness()
        pso = PSO(popu, num_particles, maxiter, p_size, classes, class_mlp, test_set, "class", shape1, shape2, shape3)
        loss = pso.loss_best
        # print(f'{df_name} Layers PSO Test {i} performance: Accuracy-> {loss["Accuracy"]}, Precision-> {loss["Precision"]}, Recall-> {loss["Recall"]}, F1-Score-> {loss["F1"]}\n')
        avg_accuracy += loss["Accuracy"]
        avg_F1 += loss["F1"]
        avg_precision += loss["Precision"]
        avg_recall += loss["Recall"]
        i += 1
    print(f'{df_name} Layers PSO Average Performance: Accuracy-> {avg_accuracy/10}, Precision-> {avg_precision/10}, Recall-> {avg_recall/10}, F1-Score-> {avg_F1/10}\n\n')

def pso_regression_results(shape1, shape2 = None, shape3 = None):
    pass

# function to print all necessary output for video demonstration
def video_prints():
    # print ga operations
    # print de operations
    # print pso operations
    # show idv and avg performance of each over 10 folds of one class and one regress
    pass

######################################################################################################

# print every glass outputs
def print_glass_set():
    print("################# Processing Classification Glass Dataset #######################\n")
    glass_classes = [1,2,3,4,5,6,7]
    # intiialize glass_mlps
    glass_mlp_0 = MLP(10, [], 7)
    glass_mlp_1 = MLP(10, [6], 7)
    glass_mlp_2 = MLP(10, [8,6], 7)
    # returns test datasets
    glass_train_output, glass_test_output = MLP_HELPER.mlp_glass_data()
    glass_weight_list_0 = MLP_HELPER.get_mlp_weights(glass_mlp_0,glass_train_output,0.01,250)
    glass_weight_list_1 = MLP_HELPER.get_mlp_weights(glass_mlp_1,glass_train_output,0.01,500)
    glass_weight_list_2 = MLP_HELPER.get_mlp_weights(glass_mlp_2,glass_train_output,0.001,1000)

    ################ Glass MLP ##################
    mlp_class_results(glass_test_output, glass_weight_list_0,glass_mlp_0,glass_classes,"Glass 0")
    mlp_class_results(glass_test_output, glass_weight_list_1,glass_mlp_1,glass_classes,"Glass 1")
    mlp_class_results(glass_test_output, glass_weight_list_2,glass_mlp_2,glass_classes,"Glass 2")

    # ################ Glass GA ###################
    # create glass GA population
    p_size = 500
    ga_popu_0 = create_population(p_size,glass_weight_list_0)
    ga_popu_1 = create_population(p_size,glass_weight_list_1)
    ga_popu_2 = create_population(p_size,glass_weight_list_2)

    # # for 0 layers
    ga_class_results(glass_test_output,glass_mlp_0, ga_popu_0, num_gens=200, t_size=2, c_prob=0.9, classes = glass_classes, n_size=math.ceil(p_size*(2/3)),df_name="Glass 0")
    # # # for 1 layers
    ga_class_results(glass_test_output,glass_mlp_1, ga_popu_1, num_gens=200, t_size=2, c_prob=0.9, classes = glass_classes, n_size=math.ceil(p_size*(2/3)),df_name="Glass 1")
    # # # for 2 layers
    ga_class_results(glass_test_output,glass_mlp_2, ga_popu_2, num_gens=200, t_size=2, c_prob=0.9, classes = glass_classes, n_size=math.ceil(p_size*(2/3)),df_name="Glass 2")

    # ############### Glass DE ####################
    # create glass DE population
    p_size = 200
    de_popu_0 = create_population(p_size,glass_weight_list_0)
    de_popu_1 = create_population(p_size,glass_weight_list_1)
    de_popu_2 = create_population(p_size,glass_weight_list_2)
    num_generations_de = 200

    # for 0 layers
    de_class_results(glass_test_output,glass_mlp_0, de_popu_0, num_generations_de, glass_classes, "Glass 0",0.2)
    # for 1 layers
    de_class_results(glass_test_output,glass_mlp_1, de_popu_1, num_generations_de, glass_classes, "Glass 1",0.5)
    # for 2 layers
    de_class_results(glass_test_output,glass_mlp_2, de_popu_2, num_generations_de, glass_classes, "Glass 2",0.7)

    # ############### GLass PSO ####################
    p_size = 20
    all_popu_0 = create_population(p_size,glass_weight_list_0)
    all_popu_1 = create_population(p_size,glass_weight_list_1)
    all_popu_2 = create_population(p_size,glass_weight_list_2)
    num_particles = 10 
    maxiter = 30

    # # for 0 layers
    pso_class_results(glass_test_output, glass_mlp_0, all_popu_0, glass_classes, "Glass 0", p_size, num_particles, maxiter, shape1=all_popu_0[0][0].shape)
    # # for 1 layers
    pso_class_results(glass_test_output, glass_mlp_1, all_popu_1, glass_classes, "Glass 1", p_size, num_particles, maxiter, shape1=all_popu_1[0][0].shape, shape2=all_popu_1[0][1].shape)
    # # for 2 layers
    pso_class_results(glass_test_output, glass_mlp_2, all_popu_2, glass_classes, "Glass 2", p_size, num_particles, maxiter, shape1=all_popu_2[0][0].shape, shape2=all_popu_2[0][1].shape, shape3=all_popu_2[0][2].shape)

# print every soy outputs
def print_soy_set():
    print("################# Processing Classification Soybean Dataset #######################")
    soy_classes = [1,2,3,4]
    soy_mlp_0 = MLP(22, [], 4)
    soy_mlp_1 = MLP(22, [6], 4)
    soy_mlp_2 = MLP(22, [6,6], 4)

    # returns test datasets
    soy_train_output, soy_test_output = MLP_HELPER.mlp_soybean_data()
    soy_weight_list_0 = MLP_HELPER.get_mlp_weights(soy_mlp_0,soy_train_output,0.01,250)
    soy_weight_list_1 = MLP_HELPER.get_mlp_weights(soy_mlp_1,soy_train_output,0.01,250)
    soy_weight_list_2 = MLP_HELPER.get_mlp_weights(soy_mlp_2,soy_train_output,0.01,250)

    # ################ Soy MLP ##################
    mlp_class_results(soy_test_output, soy_weight_list_0,soy_mlp_0,soy_classes,"Soy 0")
    mlp_class_results(soy_test_output, soy_weight_list_1,soy_mlp_1,soy_classes,"Soy 1")
    mlp_class_results(soy_test_output, soy_weight_list_2,soy_mlp_2,soy_classes,"Soy 2")

    # ################ Soy GA ###################
    # create soy GA population
    p_size = 200
    ga_popu_0 = create_population(p_size,soy_weight_list_0)
    ga_popu_1 = create_population(p_size,soy_weight_list_1)
    ga_popu_2 = create_population(p_size,soy_weight_list_2)

    # for 0 layers
    ga_class_results(soy_test_output,soy_mlp_0, ga_popu_0, num_gens=200, t_size=2, c_prob=0.9, classes = soy_classes, n_size=math.ceil(p_size*(2/3)),df_name="Soy 0")
    # for 1 layers
    ga_class_results(soy_test_output,soy_mlp_1, ga_popu_1, num_gens=200, t_size=2, c_prob=0.9, classes = soy_classes, n_size=math.ceil(p_size*(2/3)),df_name="Soy 1")
    # for 2 layers
    ga_class_results(soy_test_output,soy_mlp_2, ga_popu_2, num_gens=200, t_size=2, c_prob=0.9, classes = soy_classes, n_size=math.ceil(p_size*(2/3)),df_name="Soy 2")

    # ############### Soy DE ####################
    # create soy GA population
    p_size = 200
    de_popu_0 = create_population(p_size,soy_weight_list_0)
    de_popu_1 = create_population(p_size,soy_weight_list_1)
    de_popu_2 = create_population(p_size,soy_weight_list_2)
    num_generations_de = 200

    # for 0 layers
    de_class_results(soy_test_output,soy_mlp_0, de_popu_0, num_generations_de, soy_classes, "Soy 2")
    # for 1 layer
    de_class_results(soy_test_output,soy_mlp_1, de_popu_1, num_generations_de, soy_classes, "Soy 2")
    # for 2 layers
    de_class_results(soy_test_output,soy_mlp_2, de_popu_2, num_generations_de, soy_classes, "Soy 2")

    # ############### Soy PSO ####################
    p_size = 20
    all_popu_0 = create_population(p_size,soy_weight_list_0)
    all_popu_1 = create_population(p_size,soy_weight_list_1)
    all_popu_2 = create_population(p_size,soy_weight_list_2)
    num_particles = 10 
    maxiter = 30
    
    # for 0 layers
    pso_class_results(soy_test_output[0:5], soy_mlp_0, all_popu_0, soy_classes, "Soy 0", p_size, num_particles, maxiter, shape1=all_popu_0[0][0].shape)
    # for 1 layer
    pso_class_results(soy_test_output[0:5], soy_mlp_1, all_popu_1, soy_classes, "Soy 0", p_size, num_particles, maxiter, shape1=all_popu_1[0][0].shape, shape2=all_popu_1[0][1].shape)
    # for 2 layers
    pso_class_results(soy_test_output[0:5], soy_mlp_2, all_popu_2, soy_classes, "Soy 0", p_size, num_particles, maxiter, shape1=all_popu_2[0][0].shape, shape2=all_popu_2[0][1].shape, shape3=all_popu_2[0][2].shape)
   
# print every soy outputs
def print_cancer_set():
    cancer_classes = [2,4]
    print("################# Processing Classification Cancer Dataset #######################")
    cancer_mlp_0 = MLP(10, [], 2)
    cancer_mlp_1 = MLP(10, [6], 2)
    cancer_mlp_2 = MLP(10, [6,6], 2)

    # returns test datasets
    cancer_train_output, cancer_test_output = MLP_HELPER.mlp_cancer_data()
    cancer_weight_list_0 = MLP_HELPER.get_mlp_weights(cancer_mlp_0,cancer_train_output,0.01,250)
    cancer_weight_list_1 = MLP_HELPER.get_mlp_weights(cancer_mlp_1,cancer_train_output,0.01,250)
    cancer_weight_list_2 = MLP_HELPER.get_mlp_weights(cancer_mlp_2,cancer_train_output,0.01,250)

    ################ Cancer MLP ##################
    mlp_class_results(cancer_test_output, cancer_weight_list_0,cancer_mlp_0,cancer_classes,"Cancer 0")
    mlp_class_results(cancer_test_output, cancer_weight_list_1,cancer_mlp_1,cancer_classes,"Cancer 1")
    mlp_class_results(cancer_test_output, cancer_weight_list_2,cancer_mlp_2,cancer_classes,"Cancer 2")

    ################ Cancer GA ###################
    # create cancer GA population
    p_size = 200
    ga_popu_0 = create_population(p_size,cancer_weight_list_0)
    ga_popu_1 = create_population(p_size,cancer_weight_list_1)
    ga_popu_2 = create_population(p_size,cancer_weight_list_2)

    # for 0 layers
    ga_class_results(cancer_test_output,cancer_mlp_0, ga_popu_0, num_gens=200, t_size=2, c_prob=0.9, classes = cancer_classes, n_size=math.ceil(p_size*(2/3)),df_name="Cancer 0")
    # for 1 layers
    ga_class_results(cancer_test_output,cancer_mlp_1, ga_popu_1, num_gens=200, t_size=2, c_prob=0.9, classes = cancer_classes, n_size=math.ceil(p_size*(2/3)),df_name="Cancer 1")
    # for 2 layers
    ga_class_results(cancer_test_output,cancer_mlp_2, ga_popu_2, num_gens=200, t_size=2, c_prob=0.9, classes = cancer_classes, n_size=math.ceil(p_size*(2/3)),df_name="Cancer 2")

    ############### Cancer DE ####################
    # create cancer DE population
    p_size = 200
    de_popu_0 = create_population(p_size,cancer_weight_list_0)
    de_popu_1 = create_population(p_size,cancer_weight_list_1)
    de_popu_2 = create_population(p_size,cancer_weight_list_2)
    num_generations_de = 200

    # for 0 layers
    de_class_results(cancer_test_output,cancer_mlp_0, de_popu_0, num_generations_de, cancer_classes, "Cancer 2")
    # for 1 layer
    de_class_results(cancer_test_output,cancer_mlp_1, de_popu_1, num_generations_de, cancer_classes, "Cancer 2")
    # for 2 layers
    de_class_results(cancer_test_output,cancer_mlp_2, de_popu_2, num_generations_de, cancer_classes, "Cancer 2")

    # ############### Cancer PSO ####################
    p_size = 20
    all_popu_0 = create_population(p_size,cancer_weight_list_0)
    all_popu_1 = create_population(p_size,cancer_weight_list_1)
    all_popu_2 = create_population(p_size,cancer_weight_list_2)
    num_particles = 10 
    maxiter = 30
    
    # for 0 layers
    pso_class_results(cancer_test_output[0:5], cancer_mlp_0, all_popu_0, cancer_classes, "Cancer 0", p_size, num_particles, maxiter, shape1=all_popu_0[0][0].shape)
    # for 1 layer
    pso_class_results(cancer_test_output[0:5], cancer_mlp_1, all_popu_1, cancer_classes, "Cancer 0", p_size, num_particles, maxiter, shape1=all_popu_1[0][0].shape, shape2=all_popu_1[0][1].shape)
    # for 2 layers
    pso_class_results(cancer_test_output[0:5], cancer_mlp_2, all_popu_2, cancer_classes, "Cancer 0", p_size, num_particles, maxiter, shape1=all_popu_2[0][0].shape, shape2=all_popu_2[0][1].shape, shape3=all_popu_2[0][2].shape)

# print every machine outputs
def print_machine_set():
    print("################# Processing Classification Machine Dataset #######################")
    p_size = 200
    machine_mlp_0 = MLP(8, [], 1)
    machine_mlp_1 = MLP(8, [10], 1)
    machine_mlp_2 = MLP(8, [10,10], 1)

    # returns test datasets
    machine_train_output, machine_test_output, x_max, x_min = MLP_HELPER.mlp_machine_data()
    machine_weight_list_0 = MLP_HELPER.get_mlp_weights(machine_mlp_0,machine_train_output,0.01,250)
    machine_weight_list_1 = MLP_HELPER.get_mlp_weights(machine_mlp_1,machine_train_output,0.01,250)
    machine_weight_list_2 = MLP_HELPER.get_mlp_weights(machine_mlp_2,machine_train_output,0.01,250)

    # ################ Machine MLP #################
    mlp_regress_results(machine_test_output, machine_weight_list_0,machine_mlp_0, x_min, x_max,"Machine 0")
    mlp_regress_results(machine_test_output, machine_weight_list_1,machine_mlp_1, x_min, x_max,"Machine 1")
    mlp_regress_results(machine_test_output, machine_weight_list_2,machine_mlp_2, x_min, x_max,"Machine 2")
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
    # create machine GA population
    p_size = 500
    ga_popu_0 = create_population(p_size,machine_weight_list_0)
    ga_popu_1 = create_population(p_size,machine_weight_list_1)
    ga_popu_2 = create_population(p_size,machine_weight_list_2)

    # for 0 layers
    ga_regress_results(machine_test_output,machine_mlp_0, ga_popu_0, num_gens=200, t_size=2, c_prob=0.9, n_size=math.ceil(p_size*(2/3)), x_max=x_max, x_min=x_min, df_name="Machine 0")
    # for 1 layers
    ga_regress_results(machine_test_output,machine_mlp_1, ga_popu_1, num_gens=200, t_size=2, c_prob=0.9,  n_size=math.ceil(p_size*(2/3)), x_max=x_max, x_min=x_min, df_name="Machine 1")
    # for 2 layers
    ga_regress_results(machine_test_output,machine_mlp_2, ga_popu_2, num_gens=200, t_size=2, c_prob=0.9,  n_size=math.ceil(p_size*(2/3)), x_max=x_max, x_min=x_min, df_name="Machine 2")
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
    # create machine DE population
    p_size = 200
    de_popu_0 = create_population(p_size,machine_weight_list_0)
    de_popu_1 = create_population(p_size,machine_weight_list_1)
    de_popu_2 = create_population(p_size,machine_weight_list_2)
    num_generations_de = 200

    # for 0 layers
    de_regress_results(machine_test_output, machine_mlp_0, de_popu_0, num_generations_de,x_max,x_min,"Machine 0",scale_factor=0.5)
    # for 1 layer
    de_regress_results(machine_test_output, machine_mlp_1, de_popu_1, num_generations_de,x_max,x_min,"Machine 0",scale_factor=0.5)
    # for 2 layers
    de_regress_results(machine_test_output, machine_mlp_2, de_popu_2, num_generations_de,x_max,x_min,"Machine 0",scale_factor=0.5)
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
    
# print every forestfire outputs
def print_forestfires_set():
    print("################# Processing Regression Forestfire Dataset #######################")
    p_size = 200
    forestfires_mlp_0 = MLP(13, [], 1)
    forestfires_mlp_1 = MLP(13, [6], 1)
    forestfires_mlp_2 = MLP(13, [6,6], 1)

    # returns test datasets
    forestfires_train_output, forestfires_test_output, x_max, x_min = MLP_HELPER.mlp_forestfires_data()
    forestfires_weight_list_0 = MLP_HELPER.get_mlp_weights(forestfires_mlp_0,forestfires_train_output,0.01,250)
    forestfires_weight_list_1 = MLP_HELPER.get_mlp_weights(forestfires_mlp_1,forestfires_train_output,0.01,250)
    forestfires_weight_list_2 = MLP_HELPER.get_mlp_weights(forestfires_mlp_2,forestfires_train_output,0.01,250)

    # ################ Forest Fire MLP #################
    mlp_regress_results(forestfires_test_output, forestfires_weight_list_0,forestfires_mlp_0, x_min, x_max,"Forestfires 0")
    mlp_regress_results(forestfires_test_output, forestfires_weight_list_1,forestfires_mlp_1, x_min, x_max,"Forestfires 1")
    mlp_regress_results(forestfires_test_output, forestfires_weight_list_2,forestfires_mlp_2, x_min, x_max,"Forestfires 2")

    # ################ Forest Fire GA ###################
    # create forestfires GA population
    p_size = 500
    ga_popu_0 = create_population(p_size,forestfires_weight_list_0)
    ga_popu_1 = create_population(p_size,forestfires_weight_list_1)
    ga_popu_2 = create_population(p_size,forestfires_weight_list_2)

    # for 0 layers
    ga_regress_results(forestfires_test_output,forestfires_mlp_0, ga_popu_0, num_gens=200, t_size=2, c_prob=0.9, n_size=math.ceil(p_size*(2/3)), x_max=x_max, x_min=x_min, df_name="Forestfires 0")
    # for 1 layers
    ga_regress_results(forestfires_test_output,forestfires_mlp_1, ga_popu_1, num_gens=200, t_size=2, c_prob=0.9, n_size=math.ceil(p_size*(2/3)), x_max=x_max, x_min=x_min, df_name="Forestfires 1")
    # for 2 layers
    ga_regress_results(forestfires_test_output,forestfires_mlp_2, ga_popu_2, num_gens=200, t_size=2, c_prob=0.9, n_size=math.ceil(p_size*(2/3)), x_max=x_max, x_min=x_min, df_name="Forestfires 2")

    # ################ Forest Fire DE ####################
    p_size = 200
    de_popu_0 = create_population(p_size,forestfires_weight_list_0)
    de_popu_1 = create_population(p_size,forestfires_weight_list_1)
    de_popu_2 = create_population(p_size,forestfires_weight_list_2)
    num_generations_de = 200

    # for 0 layers
    de_regress_results(forestfires_test_output, forestfires_mlp_0, de_popu_0, num_generations_de,x_max,x_min,"Forestfires 0",scale_factor=0.5)
    # for 1 layer
    de_regress_results(forestfires_test_output, forestfires_mlp_1, de_popu_1, num_generations_de,x_max,x_min,"Forestfires 0",scale_factor=0.5)
    # for 2 layers
    de_regress_results(forestfires_test_output, forestfires_mlp_2, de_popu_2, num_generations_de,x_max,x_min,"Forestfires 0",scale_factor=0.5)

    # ################ Forest Fire PSO ####################

# print every abalone outputs
def print_abalone_set():
    print("################# Processing Classification Abalone Dataset #######################")
    p_size = 200
    abalone_mlp_0 = MLP(10, [], 1)
    abalone_mlp_1 = MLP(10, [6], 1)
    abalone_mlp_2 = MLP(10, [6,6], 1)

    # returns test datasets
    abalone_train_output, abalone_test_output, x_max, x_min = MLP_HELPER.mlp_abalone_data()
    abalone_weight_list_0 = MLP_HELPER.get_mlp_weights(abalone_mlp_0,abalone_train_output,0.01,250)
    abalone_weight_list_1 = MLP_HELPER.get_mlp_weights(abalone_mlp_1,abalone_train_output,0.01,250)
    abalone_weight_list_2 = MLP_HELPER.get_mlp_weights(abalone_mlp_2,abalone_train_output,0.01,250)

    # ################ Abalone MLP #################
    mlp_regress_results(abalone_test_output, abalone_weight_list_0,abalone_mlp_0, x_min, x_max,"Abalone 0")
    mlp_regress_results(abalone_test_output, abalone_weight_list_1,abalone_mlp_1, x_min, x_max,"Abalone 1")
    mlp_regress_results(abalone_test_output, abalone_weight_list_2,abalone_mlp_2, x_min, x_max,"Abalone 2")

    # ################ Abalone GA ###################
    p_size = 500
    ga_popu_0 = create_population(p_size,abalone_weight_list_0)
    ga_popu_1 = create_population(p_size,abalone_weight_list_1)
    ga_popu_2 = create_population(p_size,abalone_weight_list_2)

    # for 0 layers
    ga_regress_results(abalone_test_output,abalone_mlp_0, ga_popu_0, num_gens=200, t_size=2, c_prob=0.9, n_size=math.ceil(p_size*(2/3)), x_max=x_max, x_min=x_min, df_name="Abalone 0")
    # for 1 layers
    ga_regress_results(abalone_test_output,abalone_mlp_1, ga_popu_1, num_gens=200, t_size=2, c_prob=0.9, n_size=math.ceil(p_size*(2/3)), x_max=x_max, x_min=x_min, df_name="Abalone 1")
    # for 2 layers
    ga_regress_results(abalone_test_output,abalone_mlp_2, ga_popu_2, num_gens=200, t_size=2, c_prob=0.9, n_size=math.ceil(p_size*(2/3)), x_max=x_max, x_min=x_min, df_name="Abalone 2")

    # ################ Abalone DE ####################
    p_size = 200
    de_popu_0 = create_population(p_size,abalone_weight_list_0)
    de_popu_1 = create_population(p_size,abalone_weight_list_1)
    de_popu_2 = create_population(p_size,abalone_weight_list_2)
    num_generations_de = 200

    # for 0 layers
    de_regress_results(abalone_test_output, abalone_mlp_0, de_popu_0, num_generations_de,x_max,x_min,"Abalone 0",scale_factor=0.5)
    # for 1 layer
    de_regress_results(abalone_test_output, abalone_mlp_1, de_popu_1, num_generations_de,x_max,x_min,"Abalone 0",scale_factor=0.5)
    # for 2 layers
    de_regress_results(abalone_test_output, abalone_mlp_2, de_popu_2, num_generations_de,x_max,x_min,"Abalone 0",scale_factor=0.5)

    # ################ Abalone PSO ####################

if __name__ == "__main__":

    ############  Classification Instances  #####################################

    print_glass_set()
    print_soy_set()
    print_cancer_set()

    # ################  Regression Instances  #####################################
    print_machine_set()
    print_forestfires_set()
    print_abalone_set()