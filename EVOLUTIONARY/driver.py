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
        loss = UTILS.calculate_loss_np(UTILS, result, classes, test_set['class'].values)
        # print(f'{df_name} Layer MLP {i} performance: Accuracy-> {loss["Accuracy"]}, Precision-> {loss["Precision"]}, Recall-> {loss["Recall"]}, F1-Score-> {loss["F1"]}\n')
        avg_accuracy += loss["Accuracy"]
        avg_F1 += loss["F1"]
        avg_precision += loss["Precision"]
        avg_recall += loss["Recall"]
        i += 1
    print(f'{df_name} Layer MLP Average Performance: Accuracy-> {avg_accuracy/10}, Precision-> {avg_precision/10}, Recall-> {avg_recall/10}, F1-Score-> {avg_F1/10}\n\n')

# function to print results for regression mlps
def mlp_regress_results(test_output, weight_list, regress_mlp, x_min, x_max, df_name, prevent=False):
    i = 1
    avg_MAE, avg_MSE, avg_MdAE, avg_MAPE = 0,0,0,0
    for (test_set, weights) in zip(test_output, weight_list):
        result = UTILS.get_performance(UTILS, regress_mlp, weights, classes=None, input=test_set)
        loss = UTILS.calculate_loss_for_regression(UTILS, result, test_set, x_max, x_min, prevent)
        # print(f'{df_name} MLP Test {i} performance: MAE-> {loss["MAE"]}, MSE-> {loss["MSE"]}, MdAE-> {loss["MdAE"]}, MAPE-> {loss["MAPE"]}\n')
        avg_MAE += loss["MAE"]
        avg_MSE += loss["MSE"]
        avg_MdAE += loss["MdAE"]
        if not prevent:
            avg_MAPE += loss["MAPE"]
        else:
            avg_MAPE += 0
        i += 1
    print(f'{df_name} MLP Average Performance: MAE-> {avg_MAE/10}, MSE-> {avg_MSE/10}, MdAE-> {avg_MdAE/10}, MAPE-> {avg_MAPE/10}\n\n')

# function to print results for class ga's
def ga_class_results(test_output, class_mlp, popu, num_gens, t_size, c_prob, classes, n_size, df_name, verbose=False):
    i = 1
    avg_accuracy, avg_F1, avg_precision, avg_recall = 0,0,0,0
    for test_set in test_output:
        ga = GA("class", class_mlp, popu, num_generations = num_gens, tournament_size = t_size, crossover_probability=c_prob, classes=classes, n_size=n_size, test_values=test_set, verbose=verbose)
        loss = ga.run()
        # print(f'{df_name} Layer GA Test {i} performance: Accuracy-> {loss["Accuracy"]}, Precision-> {loss["Precision"]}, Recall-> {loss["Recall"]}, F1-Score-> {loss["F1"]}\n')
        avg_accuracy += loss["Accuracy"]
        avg_F1 += loss["F1"]
        avg_precision += loss["Precision"]
        avg_recall += loss["Recall"]
        i += 1
    print(f'{df_name} Layer GA Average Performance: Accuracy-> {avg_accuracy/10}, Precision-> {avg_precision/10}, Recall-> {avg_recall/10}, F1-Score-> {avg_F1/10}\n\n')

# function to print results for regression ga's
def ga_regress_results(test_output, regress_mlp, popu, num_gens, t_size, c_prob, n_size, x_max=None, x_min=None, df_name="Name", prevent=False):
    i = 1
    avg_MAE, avg_MSE, avg_MdAE, avg_MAPE = 0,0,0,0
    for test_set in test_output:
        ga = GA("regress", regress_mlp, popu, num_generations=num_gens, tournament_size=t_size, crossover_probability=c_prob, n_size=n_size, test_values=test_set, x_max=x_max, x_min=x_min, prevent=prevent)
        loss = ga.run()
        # print(f'{df_name} GA Test {i} performance: MAE-> {loss["MAE"]}, MSE-> {loss["MSE"]}, MdAE-> {loss["MdAE"]}, MAPE-> {loss["MAPE"]}\n')
        avg_MAE += loss["MAE"]
        avg_MSE += loss["MSE"]
        avg_MdAE += loss["MdAE"]
        if not prevent:
            avg_MAPE += loss["MAPE"]
        else:
            avg_MAPE += 0
        i += 1
    print(f'{df_name} GA Average Performance: MAE-> {avg_MAE/10}, MSE-> {avg_MSE/10}, MdAE-> {avg_MdAE/10}, MAPE-> {avg_MAPE/10}\n\n')

# function to print results for class de's
def de_class_results(test_output, class_mlp, popu, num_generations, classes, df_name, scale_factor, verbose=False):
    i = 1
    avg_accuracy, avg_F1, avg_precision, avg_recall = 0,0,0,0
    for test_set in test_output:
        de = DE("class",class_mlp,popu,num_generations=num_generations,classes=classes, test_values=test_set, scale_factor=scale_factor, verbose=verbose)
        loss = de.run()
        # print(f'{df_name} Layer DE Test {i} performance: Accuracy-> {loss["Accuracy"]}, Precision-> {loss["Precision"]}, Recall-> {loss["Recall"]}, F1-Score-> {loss["F1"]}\n')
        avg_accuracy += loss["Accuracy"]
        avg_F1 += loss["F1"]
        avg_precision += loss["Precision"]
        avg_recall += loss["Recall"]
        i += 1
    print(f'{df_name} Layer DE Average Performance: Accuracy-> {avg_accuracy/10}, Precision-> {avg_precision/10}, Recall-> {avg_recall/10}, F1-Score-> {avg_F1/10}\n\n')

# function to print results for de's
def de_regress_results(test_output, regress_mlp, popu, num_generations, x_max, x_min, df_name, scale_factor, prevent = False):
    i = 1
    avg_MAE, avg_MSE, avg_MdAE, avg_MAPE = 0,0,0,0
    for test_set in test_output:
        de = DE("regress",regress_mlp,popu,num_generations=num_generations, test_values=test_set, x_max=x_max, x_min=x_min, scale_factor=scale_factor, prevent=prevent)
        loss = de.run()
        # print(f'{df_name} DE Test {i} performance: MAE-> {loss["MAE"]}, MSE-> {loss["MSE"]}, MdAE-> {loss["MdAE"]}, MAPE-> {loss["MAPE"]}\n')
        avg_MAE += loss["MAE"]
        avg_MSE += loss["MSE"]
        avg_MdAE += loss["MdAE"]
        if not prevent:
            avg_MAPE += loss["MAPE"]
        else:
            avg_MAPE += 0
        i += 1
    print(f'{df_name} DE Average Performance: MAE-> {avg_MAE/10}, MSE-> {avg_MSE/10}, MdAE-> {avg_MdAE/10}, MAPE-> {avg_MAPE/10}\n\n')

def pso_class_results(test_output, class_mlp, popu, classes, df_name, p_size, num_particles, maxiter, shape1, shape2 = None, shape3 = None, verbose=False):
    i = 1
    avg_accuracy, avg_F1, avg_precision, avg_recall = 0,0,0,0
    for test_set in test_output:
        part = Particle(popu, p_size, classes, class_mlp, test_set, "class", shape1, shape2, shape3, verbose=verbose)
        part.fitness()
        pso = PSO(popu, num_particles, maxiter, p_size, classes, class_mlp, test_set, "class", shape1, shape2, shape3, verbose=verbose)
        loss = pso.loss_best
        # print(f'{df_name} Layers PSO Test {i} performance: Accuracy-> {loss["Accuracy"]}, Precision-> {loss["Precision"]}, Recall-> {loss["Recall"]}, F1-Score-> {loss["F1"]}\n')
        avg_accuracy += loss["Accuracy"]
        avg_F1 += loss["F1"]
        avg_precision += loss["Precision"]
        avg_recall += loss["Recall"]
        i += 1
    print(f'{df_name} Layers PSO Average Performance: Accuracy-> {avg_accuracy/10}, Precision-> {avg_precision/10}, Recall-> {avg_recall/10}, F1-Score-> {avg_F1/10}\n\n')

def pso_regress_results(test_output, regress_mlp, popu, df_name, p_size, num_particles, maxiter, shape1, shape2 = None, shape3 = None, prevent = False):
    i = 1
    avg_MAE, avg_MSE, avg_MdAE, avg_MAPE = 0,0,0,0
    for test_set in test_output:
        part = Particle(popu, p_size, None, regress_mlp, test_set, "regress", shape1, shape2, shape3)
        part.fitness()
        pso = PSO(popu, num_particles, maxiter, p_size, None, regress_mlp, test_set, "regress", shape1, shape2, shape3)
        loss = pso.loss_best
        # print(f'{df_name} Layers PSO Test {i} performance: Accuracy-> {loss["Accuracy"]}, Precision-> {loss["Precision"]}, Recall-> {loss["Recall"]}, F1-Score-> {loss["F1"]}\n')
        avg_MAE += loss["MAE"]
        avg_MSE += loss["MSE"]
        avg_MdAE += loss["MdAE"]
        if not prevent:
            avg_MAPE += loss["MAPE"]
        else:
            avg_MAPE += 0
        i += 1
    print(f'{df_name} PSO Average Performance: MAE-> {avg_MAE/10}, MSE-> {avg_MSE/10}, MdAE-> {avg_MdAE/10}, MAPE-> {avg_MAPE/10}\n\n')

###################################### Print Functions ################################################################

# function to print all necessary output for video demonstration
def video_prints():
    # show avg performance of each over 10 folds of one class and one regress for all network types
    print("################# Processing Classification Glass Dataset #######################\n")
    glass_classes = [1,2,3,4,5,6,7]
    # intiialize glass_mlps
    glass_mlp_0 = MLP(10, [], 7)
    glass_mlp_1 = MLP(10, [6], 7)
    glass_mlp_2 = MLP(10, [8,6], 7)
    # returns test datasets
    glass_train_output, glass_test_output = MLP_HELPER.mlp_glass_data()
    glass_weight_list_0 = MLP_HELPER.get_mlp_weights(glass_mlp_0,glass_train_output,0.01,250)
    glass_weight_list_1 = MLP_HELPER.get_mlp_weights(glass_mlp_1,glass_train_output,0.01,1000)
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
    ga_class_results(glass_test_output,glass_mlp_0, ga_popu_0[0:200], num_gens=200, t_size=2, c_prob=0.9, classes = glass_classes, n_size=math.ceil(200*(2/3)),df_name="Glass 0 w/ p_size 200, gen 200, c_prob 0.9")
    # # # for 1 layers
    ga_class_results(glass_test_output,glass_mlp_1, ga_popu_1[0:200], num_gens=200, t_size=2, c_prob=0.9, classes = glass_classes, n_size=math.ceil(200*(2/3)),df_name="Glass 1 w/ p_size 200, gen 200, c_prob 0.9")
    # # # for 2 layers
    ga_class_results(glass_test_output,glass_mlp_2, ga_popu_2[0:200], num_gens=200, t_size=2, c_prob=0.9, classes = glass_classes, n_size=math.ceil(200*(2/3)),df_name="Glass 2 w/ p_size 200, gen 200, c_prob 0.9")

    # ############### Glass DE ####################
    # create glass DE population
    p_size = 500
    de_popu_0 = create_population(p_size,glass_weight_list_0)
    de_popu_1 = create_population(p_size,glass_weight_list_1)
    de_popu_2 = create_population(p_size,glass_weight_list_2)
    num_generations_de = 200

    # for 0 layers
    de_class_results(glass_test_output,glass_mlp_0, de_popu_0[0:200], num_generations_de, glass_classes, "Glass 0 p_size = 200, generations 200, scale_factor 0.2",0.2)
    # for 1 layers
    de_class_results(glass_test_output,glass_mlp_1, de_popu_1[0:200], num_generations_de, glass_classes, "Glass 1 p_size = 200, generations 200, scale_factor 0.5",0.5)
    # for 2 layers
    de_class_results(glass_test_output,glass_mlp_2, de_popu_2[0:200], num_generations_de, glass_classes, "Glass 2 p_size = 200, generations 200, scale_factor 0.7",0.7)

    # ############### GLass PSO ####################
    p_size = 20
    all_popu_0 = create_population(p_size,glass_weight_list_0)
    all_popu_1 = create_population(p_size,glass_weight_list_1)
    all_popu_2 = create_population(p_size,glass_weight_list_2)
    num_particles = 10 
    maxiter = 30

    # # for 0 layers
    pso_class_results(glass_test_output, glass_mlp_0, all_popu_0, glass_classes, "Glass 0 num particles 10, p_size = 20, maxiter = 30, c1 is 1 , c2 is 2", p_size, num_particles, maxiter, shape1=all_popu_0[0][0].shape)
    # # for 1 layers
    pso_class_results(glass_test_output, glass_mlp_1, all_popu_1, glass_classes, "Glass 1 num particles 10, p_size = 20, maxiter = 30, c1 is 1 , c2 is 2", p_size, num_particles, maxiter, shape1=all_popu_1[0][0].shape, shape2=all_popu_1[0][1].shape)
    # # for 2 layers
    pso_class_results(glass_test_output, glass_mlp_2, all_popu_2, glass_classes, "Glass 2 num particles 10, p_size = 20, maxiter = 30, c1 is 1 , c2 is 2", p_size, num_particles, maxiter, shape1=all_popu_2[0][0].shape, shape2=all_popu_2[0][1].shape, shape3=all_popu_2[0][2].shape)
    
    print("################# Processing Regression Forestfire Dataset #######################")
    p_size = 200
    forestfires_mlp_0 = MLP(13, [], 1)
    forestfires_mlp_1 = MLP(13, [12], 1)
    forestfires_mlp_2 = MLP(13, [12,12], 1)

    # returns test datasets
    forestfires_train_output, forestfires_test_output = MLP_HELPER.mlp_forestfires_data()
    forestfires_weight_list_0 = MLP_HELPER.get_mlp_weights(forestfires_mlp_0,forestfires_train_output,0.01,250)
    forestfires_weight_list_1 = MLP_HELPER.get_mlp_weights(forestfires_mlp_1,forestfires_train_output,0.01,500)
    forestfires_weight_list_2 = MLP_HELPER.get_mlp_weights(forestfires_mlp_2,forestfires_train_output,0.01,500)

    # ################ Forest Fire MLP #################
    mlp_regress_results(forestfires_test_output, forestfires_weight_list_0,forestfires_mlp_0, x_min=None, x_max=None,df_name="Forestfires 0", prevent=True)
    mlp_regress_results(forestfires_test_output, forestfires_weight_list_1,forestfires_mlp_1, x_min=None, x_max=None,df_name="Forestfires 1", prevent=True)
    mlp_regress_results(forestfires_test_output, forestfires_weight_list_2,forestfires_mlp_2, x_min=None, x_max=None,df_name="Forestfires 2", prevent=True)

    # ################ Forest Fire GA ###################
    # create forestfires GA population
    p_size = 200
    ga_popu_0 = create_population(p_size,forestfires_weight_list_0)
    ga_popu_1 = create_population(p_size,forestfires_weight_list_1)
    ga_popu_2 = create_population(p_size,forestfires_weight_list_2)

    # for 0 layers
    ga_regress_results(forestfires_test_output,forestfires_mlp_0, ga_popu_0, num_gens=200, t_size=2, c_prob=0.9, n_size=math.ceil(p_size*(2/3)), x_min=None, x_max=None,df_name="Forestfires 0", prevent=True)
    # for 1 layers
    ga_regress_results(forestfires_test_output,forestfires_mlp_1, ga_popu_1, num_gens=200, t_size=2, c_prob=0.9, n_size=math.ceil(p_size*(2/3)), x_min=None, x_max=None,df_name="Forestfires 1", prevent=True)
    # for 2 layers
    ga_regress_results(forestfires_test_output,forestfires_mlp_2, ga_popu_2, num_gens=200, t_size=2, c_prob=0.9, n_size=math.ceil(p_size*(2/3)), x_min=None, x_max=None,df_name="Forestfires 2", prevent=True)

    # ################ Forest Fire DE ####################
    p_size = 200
    de_popu_0 = create_population(p_size,forestfires_weight_list_0)
    de_popu_1 = create_population(p_size,forestfires_weight_list_1)
    de_popu_2 = create_population(p_size,forestfires_weight_list_2)
    num_generations_de = 200

    # for 0 layers
    de_regress_results(forestfires_test_output, forestfires_mlp_0, de_popu_0, num_generations_de,x_min=None, x_max=None,df_name="Forestfires 0",scale_factor=0.5, prevent=True)
    # for 1 layer
    de_regress_results(forestfires_test_output, forestfires_mlp_1, de_popu_1, num_generations_de,x_min=None, x_max=None,df_name="Forestfires 1",scale_factor=0.5, prevent=True)
    # for 2 layers
    de_regress_results(forestfires_test_output, forestfires_mlp_2, de_popu_2, num_generations_de,x_min=None, x_max=None,df_name="Forestfires 2",scale_factor=0.5, prevent=True)

    # ################ Forest Fire PSO ####################
    p_size = 20
    all_popu_0 = create_population(p_size,forestfires_weight_list_0)
    all_popu_1 = create_population(p_size,forestfires_weight_list_1)
    all_popu_2 = create_population(p_size,forestfires_weight_list_2)
    num_particles = 10 
    maxiter = 30
    
    # for 0 layers
    pso_regress_results(forestfires_test_output, forestfires_mlp_0, all_popu_0, "Forestfires 0", p_size, num_particles, maxiter, shape1=all_popu_0[0][0].shape, prevent=True)
    # for 1 layer
    pso_regress_results(forestfires_test_output, forestfires_mlp_1, all_popu_1, "Forestfires 1", p_size, num_particles, maxiter, shape1=all_popu_1[0][0].shape, shape2=all_popu_1[0][1].shape, prevent=True)
    # for 2 layers
    pso_regress_results(forestfires_test_output, forestfires_mlp_2, all_popu_2, "Forestfires 2", p_size, num_particles, maxiter, shape1=all_popu_2[0][0].shape, shape2=all_popu_2[0][1].shape, shape3=all_popu_2[0][2].shape, prevent=True)

    ## print ga operations
    # create glass GA population
    p_size = 500
    ga_popu_2 = create_population(p_size,glass_weight_list_2)
    # # # for 2 layers
    ga_class_results(glass_test_output,glass_mlp_2, ga_popu_2[0:200], num_gens=200, t_size=2, c_prob=0.9, classes = glass_classes, n_size=math.ceil(200*(2/3)),df_name="Glass 2 w/ p_size 200, gen 200, c_prob 0.9", verbose=True)

    ## print de operations
    p_size = 500
    de_popu_2 = create_population(p_size,glass_weight_list_2)
    num_generations_de = 200
    # for 2 layers
    de_class_results(glass_test_output,glass_mlp_2, de_popu_2[0:200], num_generations_de, glass_classes, "Glass 2 p_size = 200, generations 200, scale_factor 0.7",0.7, verbose=True)

    # print pso operations
    p_size = 20
    all_popu_2 = create_population(p_size,glass_weight_list_2)
    num_particles = 10 
    maxiter = 30
    # # for 2 layers
    pso_class_results(glass_test_output, glass_mlp_2, all_popu_2, glass_classes, "Glass 2 num particles 10, p_size = 20, maxiter = 30, c1 is 1 , c2 is 2", p_size, num_particles, maxiter, shape1=all_popu_2[0][0].shape, shape2=all_popu_2[0][1].shape, shape3=all_popu_2[0][2].shape, verbose=True)

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
    glass_weight_list_1 = MLP_HELPER.get_mlp_weights(glass_mlp_1,glass_train_output,0.01,1000)
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
    ga_class_results(glass_test_output,glass_mlp_0, ga_popu_0[0:200], num_gens=200, t_size=2, c_prob=0.9, classes = glass_classes, n_size=math.ceil(200*(2/3)),df_name="Glass 0 w/ p_size 200, gen 200, c_prob 0.9")
    # # # for 1 layers
    ga_class_results(glass_test_output,glass_mlp_1, ga_popu_1[0:200], num_gens=200, t_size=2, c_prob=0.9, classes = glass_classes, n_size=math.ceil(200*(2/3)),df_name="Glass 1 w/ p_size 200, gen 200, c_prob 0.9")
    # # # for 2 layers
    ga_class_results(glass_test_output,glass_mlp_2, ga_popu_2[0:200], num_gens=200, t_size=2, c_prob=0.9, classes = glass_classes, n_size=math.ceil(200*(2/3)),df_name="Glass 2 w/ p_size 200, gen 200, c_prob 0.9")
    # # for 0 layers
    ga_class_results(glass_test_output,glass_mlp_0, ga_popu_0, num_gens=200, t_size=2, c_prob=0.9, classes = glass_classes, n_size=math.ceil(p_size*(2/3)),df_name="Glass 0 w/ p_size 500, gen 200, c_prob 0.9")
    # # # for 1 layers
    ga_class_results(glass_test_output,glass_mlp_1, ga_popu_1, num_gens=200, t_size=2, c_prob=0.9, classes = glass_classes, n_size=math.ceil(p_size*(2/3)),df_name="Glass 1 w/ p_size 500, gen 200, c_prob 0.9")
    # # # for 2 layers
    ga_class_results(glass_test_output,glass_mlp_2, ga_popu_2, num_gens=200, t_size=2, c_prob=0.9, classes = glass_classes, n_size=math.ceil(p_size*(2/3)),df_name="Glass 2 w/ p_size 500, gen 200, c_prob 0.9")
    # # for 0 layers
    ga_class_results(glass_test_output,glass_mlp_0, ga_popu_0, num_gens=50, t_size=2, c_prob=0.9, classes = glass_classes, n_size=math.ceil(p_size*(2/3)),df_name="Glass 0 w/ p_size 500, gen 50, c_prob 0.9")
    # # # for 1 layers
    ga_class_results(glass_test_output,glass_mlp_1, ga_popu_1, num_gens=50, t_size=2, c_prob=0.9, classes = glass_classes, n_size=math.ceil(p_size*(2/3)),df_name="Glass 1 w/ p_size 500, gen 50, c_prob 0.9")
    # # # for 2 layers
    ga_class_results(glass_test_output,glass_mlp_2, ga_popu_2, num_gens=50, t_size=2, c_prob=0.9, classes = glass_classes, n_size=math.ceil(p_size*(2/3)),df_name="Glass 2 w/ p_size 500, gen 50, c_prob 0.9")
    # # for 0 layers
    ga_class_results(glass_test_output,glass_mlp_0, ga_popu_0, num_gens=400, t_size=2, c_prob=0.9, classes = glass_classes, n_size=math.ceil(p_size*(2/3)),df_name="Glass 0 w/ p_size 500, gen 400, c_prob 0.9")
    # # # for 1 layers
    ga_class_results(glass_test_output,glass_mlp_1, ga_popu_1, num_gens=400, t_size=2, c_prob=0.9, classes = glass_classes, n_size=math.ceil(p_size*(2/3)),df_name="Glass 1 w/ p_size 500, gen 400, c_prob 0.9")
    # # # for 2 layers
    ga_class_results(glass_test_output,glass_mlp_2, ga_popu_2, num_gens=400, t_size=2, c_prob=0.9, classes = glass_classes, n_size=math.ceil(p_size*(2/3)),df_name="Glass 2 w/ p_size 500, gen 400, c_prob 0.9")
    # # for 0 layers
    ga_class_results(glass_test_output,glass_mlp_0, ga_popu_0, num_gens=200, t_size=2, c_prob=0.5, classes = glass_classes, n_size=math.ceil(p_size*(2/3)),df_name="Glass 0 w/ p_size 500, gen 200, c_prob 0.5")
    # # # for 1 layers
    ga_class_results(glass_test_output,glass_mlp_1, ga_popu_1, num_gens=200, t_size=2, c_prob=0.5, classes = glass_classes, n_size=math.ceil(p_size*(2/3)),df_name="Glass 1 w/ p_size 500, gen 200, c_prob 0.5")
    # # # for 2 layers
    ga_class_results(glass_test_output,glass_mlp_2, ga_popu_2, num_gens=200, t_size=2, c_prob=0.5, classes = glass_classes, n_size=math.ceil(p_size*(2/3)),df_name="Glass 2 w/ p_size 500, gen 200, c_prob 0.5")
    # # for 0 layers
    ga_class_results(glass_test_output,glass_mlp_0, ga_popu_0, num_gens=200, t_size=2, c_prob=0.2, classes = glass_classes, n_size=math.ceil(p_size*(2/3)),df_name="Glass 0 w/ p_size 500, gen 200, c_prob 0.2")
    # # # for 1 layers
    ga_class_results(glass_test_output,glass_mlp_1, ga_popu_1, num_gens=200, t_size=2, c_prob=0.2, classes = glass_classes, n_size=math.ceil(p_size*(2/3)),df_name="Glass 1 w/ p_size 500, gen 200, c_prob 0.2")
    # # # for 2 layers
    ga_class_results(glass_test_output,glass_mlp_2, ga_popu_2, num_gens=200, t_size=2, c_prob=0.2, classes = glass_classes, n_size=math.ceil(p_size*(2/3)),df_name="Glass 2 w/ p_size 500, gen 200, c_prob 0.2")

    # ############### Glass DE ####################
    # create glass DE population
    p_size = 500
    de_popu_0 = create_population(p_size,glass_weight_list_0)
    de_popu_1 = create_population(p_size,glass_weight_list_1)
    de_popu_2 = create_population(p_size,glass_weight_list_2)
    num_generations_de = 200

    # for 0 layers
    de_class_results(glass_test_output,glass_mlp_0, de_popu_0[0:200], num_generations_de, glass_classes, "Glass 0 p_size = 200, generations 200, scale_factor 0.2",0.2)
    # for 1 layers
    de_class_results(glass_test_output,glass_mlp_1, de_popu_1[0:200], num_generations_de, glass_classes, "Glass 1 p_size = 200, generations 200, scale_factor 0.5",0.5)
    # for 2 layers
    de_class_results(glass_test_output,glass_mlp_2, de_popu_2[0:200], num_generations_de, glass_classes, "Glass 2 p_size = 200, generations 200, scale_factor 0.7",0.7)
    # for 0 layers
    de_class_results(glass_test_output,glass_mlp_0, de_popu_0[0:200], num_generations_de, glass_classes, "Glass 0 p_size = 200, generations 200, scale_factor 0.5",0.5)
    de_class_results(glass_test_output,glass_mlp_0, de_popu_0[0:200], num_generations_de, glass_classes, "Glass 0 p_size = 200, generations 200, scale_factor 0.7",0.7)
    # for 1 layers
    de_class_results(glass_test_output,glass_mlp_1, de_popu_1[0:200], num_generations_de, glass_classes, "Glass 1 p_size = 200, generations 200, scale_factor 0.2",0.2)
    de_class_results(glass_test_output,glass_mlp_1, de_popu_1[0:200], num_generations_de, glass_classes, "Glass 1 p_size = 200, generations 200, scale_factor 0.7",0.7)
    # for 2 layers
    de_class_results(glass_test_output,glass_mlp_1, de_popu_1[0:200], num_generations_de, glass_classes, "Glass 1 p_size = 200, generations 200, scale_factor 0.2",0.2)
    de_class_results(glass_test_output,glass_mlp_2, de_popu_2[0:200], num_generations_de, glass_classes, "Glass 2 p_size = 200, generations 200, scale_factor 0.5",0.5)
    # for 0 layers
    de_class_results(glass_test_output,glass_mlp_0, de_popu_0[0:50], num_generations_de, glass_classes, "Glass 0 p_size = 50, generations 200, scale_factor 0.2",0.2)
    # for 1 layers
    de_class_results(glass_test_output,glass_mlp_1, de_popu_1[0:50], num_generations_de, glass_classes, "Glass 1 p_size = 50, generations 200, scale_factor 0.5",0.5)
    # for 2 layers
    de_class_results(glass_test_output,glass_mlp_2, de_popu_2[0:50], num_generations_de, glass_classes, "Glass 2 p_size = 50, generations 200, scale_factor 0.7",0.7)
    # for 0 layers
    de_class_results(glass_test_output,glass_mlp_0, de_popu_0, num_generations_de, glass_classes, "Glass 0 p_size = 500, generations 200, scale_factor 0.2",0.2)
    # for 1 layers
    de_class_results(glass_test_output,glass_mlp_1, de_popu_1, num_generations_de, glass_classes, "Glass 1 p_size = 500, generations 200, scale_factor 0.5",0.5)
    # for 2 layers
    de_class_results(glass_test_output,glass_mlp_2, de_popu_2, num_generations_de, glass_classes, "Glass 2 p_size = 500, generations 200, scale_factor 0.7",0.7)
    # for 0 layers
    de_class_results(glass_test_output,glass_mlp_0, de_popu_0[0:200], 50, glass_classes, "Glass 0 p_size = 200, generations 200, scale_factor 0.2",0.2)
    # for 1 layers
    de_class_results(glass_test_output,glass_mlp_1, de_popu_1[0:200], 50, glass_classes, "Glass 1 p_size = 200, generations 200, scale_factor 0.5",0.5)
    # for 2 layers
    de_class_results(glass_test_output,glass_mlp_2, de_popu_2[0:200], 50, glass_classes, "Glass 2 p_size = 200, generations 50, scale_factor 0.7",0.7)
    # for 0 layers
    de_class_results(glass_test_output,glass_mlp_0, de_popu_0[0:200], 400, glass_classes, "Glass 0 p_size = 200, generations 400, scale_factor 0.2",0.2)
    # for 1 layers
    de_class_results(glass_test_output,glass_mlp_1, de_popu_1[0:200], 400, glass_classes, "Glass 1 p_size = 200, generations 400, scale_factor 0.5",0.5)
    # for 2 layers
    de_class_results(glass_test_output,glass_mlp_2, de_popu_2[0:200], 400, glass_classes, "Glass 2 p_size = 200, generations 400, scale_factor 0.7",0.7)

    # ############### GLass PSO ####################
    p_size = 20
    all_popu_0 = create_population(p_size,glass_weight_list_0)
    all_popu_1 = create_population(p_size,glass_weight_list_1)
    all_popu_2 = create_population(p_size,glass_weight_list_2)
    num_particles = 10 
    maxiter = 30

    # # for 0 layers
    pso_class_results(glass_test_output, glass_mlp_0, all_popu_0, glass_classes, "Glass 0 num particles 10, p_size = 20, maxiter = 30, c1 is 1 , c2 is 2", p_size, num_particles, maxiter, shape1=all_popu_0[0][0].shape)
    # # for 1 layers
    pso_class_results(glass_test_output, glass_mlp_1, all_popu_1, glass_classes, "Glass 1 num particles 10, p_size = 20, maxiter = 30, c1 is 1 , c2 is 2", p_size, num_particles, maxiter, shape1=all_popu_1[0][0].shape, shape2=all_popu_1[0][1].shape)
    # # for 2 layers
    pso_class_results(glass_test_output, glass_mlp_2, all_popu_2, glass_classes, "Glass 2 num particles 10, p_size = 20, maxiter = 30, c1 is 1 , c2 is 2", p_size, num_particles, maxiter, shape1=all_popu_2[0][0].shape, shape2=all_popu_2[0][1].shape, shape3=all_popu_2[0][2].shape)

    # # for 0 layers
    pso_class_results(glass_test_output, glass_mlp_0, all_popu_0, glass_classes, "Glass 0 num particles 10, p_size = 20, maxiter = 100, c1 is 1 , c2 is 2", p_size, num_particles, 100, shape1=all_popu_0[0][0].shape)
    # # for 1 layers
    pso_class_results(glass_test_output, glass_mlp_1, all_popu_1, glass_classes, "Glass 1 num particles 10, p_size = 20, maxiter = 100, c1 is 1 , c2 is 2", p_size, num_particles, 100, shape1=all_popu_1[0][0].shape, shape2=all_popu_1[0][1].shape)
    # # for 2 layers
    pso_class_results(glass_test_output, glass_mlp_2, all_popu_2, glass_classes, "Glass 2 num particles 10, p_size = 20, maxiter = 100, c1 is 1 , c2 is 2", p_size, num_particles, 100, shape1=all_popu_2[0][0].shape, shape2=all_popu_2[0][1].shape, shape3=all_popu_2[0][2].shape)

    
    p_size = 100
    all_popu_0 = create_population(p_size,glass_weight_list_0)
    all_popu_1 = create_population(p_size,glass_weight_list_1)
    all_popu_2 = create_population(p_size,glass_weight_list_2)
    num_particles = 50 
    maxiter = 30

    # # for 0 layers
    pso_class_results(glass_test_output, glass_mlp_0, all_popu_0, glass_classes, "Glass 0 num particles 50, p_size = 100, maxiter = 30, c1 is 1 , c2 is 2", p_size, num_particles, maxiter, shape1=all_popu_0[0][0].shape)
    # # for 1 layers
    pso_class_results(glass_test_output, glass_mlp_1, all_popu_1, glass_classes, "Glass 1 num particles 50, p_size = 100, maxiter = 30, c1 is 1 , c2 is 2", p_size, num_particles, maxiter, shape1=all_popu_1[0][0].shape, shape2=all_popu_1[0][1].shape)
    # # for 2 layers
    pso_class_results(glass_test_output, glass_mlp_2, all_popu_2, glass_classes, "Glass 2 num particles 50, p_size = 100, maxiter = 30, c1 is 1 , c2 is 2", p_size, num_particles, maxiter, shape1=all_popu_2[0][0].shape, shape2=all_popu_2[0][1].shape, shape3=all_popu_2[0][2].shape)

    # # for 0 layers
    pso_class_results(glass_test_output, glass_mlp_0, all_popu_0, glass_classes, "Glass 0 num particles 50, p_size = 100, maxiter = 100, c1 is 1 , c2 is 2", p_size, num_particles, 100, shape1=all_popu_0[0][0].shape)
    # # for 1 layers
    pso_class_results(glass_test_output, glass_mlp_1, all_popu_1, glass_classes, "Glass 1 num particles 50, p_size = 100, maxiter = 100, c1 is 1 , c2 is 2", p_size, num_particles, 100, shape1=all_popu_1[0][0].shape, shape2=all_popu_1[0][1].shape)
    # # for 2 layers
    pso_class_results(glass_test_output, glass_mlp_2, all_popu_2, glass_classes, "Glass 2 num particles 50, p_size = 100, maxiter = 100, c1 is 1 , c2 is 2", p_size, num_particles, 100, shape1=all_popu_2[0][0].shape, shape2=all_popu_2[0][1].shape, shape3=all_popu_2[0][2].shape)

    # # for 0 layers
    pso_class_results(glass_test_output, glass_mlp_0, all_popu_0, glass_classes, "Glass 0 num particles 50, p_size = 100, maxiter = 30, c1 is 2 , c2 is 1", p_size, num_particles, maxiter, shape1=all_popu_0[0][0].shape, c1=2,c2=1)
    # # for 1 layers
    pso_class_results(glass_test_output, glass_mlp_1, all_popu_1, glass_classes, "Glass 1 num particles 50, p_size = 100, maxiter = 30, c1 is 2 , c2 is 1", p_size, num_particles, maxiter, shape1=all_popu_1[0][0].shape, shape2=all_popu_1[0][1].shape, c1=2,c2=1)
    # # for 2 layers
    pso_class_results(glass_test_output, glass_mlp_2, all_popu_2, glass_classes, "Glass 2 num particles 50, p_size = 100, maxiter = 30, c1 is 2 , c2 is 1", p_size, num_particles, maxiter, shape1=all_popu_2[0][0].shape, shape2=all_popu_2[0][1].shape, shape3=all_popu_2[0][2].shape, c1=2,c2=1)

    # # for 0 layers
    pso_class_results(glass_test_output, glass_mlp_0, all_popu_0, glass_classes, "Glass 0 num particles 50, p_size = 100, maxiter = 30, c1 is 1 , c2 is 1", p_size, num_particles, maxiter, shape1=all_popu_0[0][0].shape, c1=1,c2=1)
    # # for 1 layers
    pso_class_results(glass_test_output, glass_mlp_1, all_popu_1, glass_classes, "Glass 1 num particles 50, p_size = 100, maxiter = 30, c1 is 1 , c2 is 1", p_size, num_particles, maxiter, shape1=all_popu_1[0][0].shape, shape2=all_popu_1[0][1].shape, c1=1,c2=1)
    # # for 2 layers
    pso_class_results(glass_test_output, glass_mlp_2, all_popu_2, glass_classes, "Glass 2 num particles 50, p_size = 100, maxiter = 30, c1 is 1 , c2 is 1", p_size, num_particles, maxiter, shape1=all_popu_2[0][0].shape, shape2=all_popu_2[0][1].shape, shape3=all_popu_2[0][2].shape, c1=1,c2=1)

# print every soy outputs
def print_soy_set():
    print("################# Processing Classification Soybean Dataset #######################")
    soy_classes = [1,2,3,4]
    soy_mlp_0 = MLP(22, [], 4)
    soy_mlp_1 = MLP(22, [18], 4)
    soy_mlp_2 = MLP(22, [22,18], 4)

    # returns test datasets
    soy_train_output, soy_test_output = MLP_HELPER.mlp_soybean_data()
    soy_weight_list_0 = MLP_HELPER.get_mlp_weights(soy_mlp_0,soy_train_output,0.01,250)
    soy_weight_list_1 = MLP_HELPER.get_mlp_weights(soy_mlp_1,soy_train_output,0.01,1000)
    soy_weight_list_2 = MLP_HELPER.get_mlp_weights(soy_mlp_2,soy_train_output,0.01,1000)
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
    de_class_results(soy_test_output,soy_mlp_0, de_popu_0, num_generations_de, soy_classes, "Soy 0")
    # for 1 layer
    de_class_results(soy_test_output,soy_mlp_1, de_popu_1, num_generations_de, soy_classes, "Soy 1")
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
    pso_class_results(soy_test_output, soy_mlp_0, all_popu_0, soy_classes, "Soy 0", p_size, num_particles, maxiter, shape1=all_popu_0[0][0].shape)
    # for 1 layer
    pso_class_results(soy_test_output, soy_mlp_1, all_popu_1, soy_classes, "Soy 1", p_size, num_particles, maxiter, shape1=all_popu_1[0][0].shape, shape2=all_popu_1[0][1].shape)
    # for 2 layers
    pso_class_results(soy_test_output, soy_mlp_2, all_popu_2, soy_classes, "Soy 2", p_size, num_particles, maxiter, shape1=all_popu_2[0][0].shape, shape2=all_popu_2[0][1].shape, shape3=all_popu_2[0][2].shape)
   
# print every cancer outputs
def print_cancer_set():
    cancer_classes = [2,4]
    print("################# Processing Classification Cancer Dataset #######################")
    cancer_mlp_0 = MLP(10, [], 2)

    # cancer_mlp_1a = MLP(10, [7], 2)

    cancer_mlp_2b = MLP(10, [7,5], 2)
    cancer_mlp_2d = MLP(10, [8,6], 2)

    # returns test datasets
    cancer_train_output, cancer_test_output = MLP_HELPER.mlp_cancer_data()
    cancer_weight_list_0 = MLP_HELPER.get_mlp_weights(cancer_mlp_0,cancer_train_output,0.01,250)
    print("0 done")
    # cancer_weight_list_1a = MLP_HELPER.get_mlp_weights(cancer_mlp_1a,cancer_train_output,0.01,1000)
    print("1 done")
    cancer_weight_list_2b = MLP_HELPER.get_mlp_weights(cancer_mlp_2b,cancer_train_output,0.001,4000)
    print("finished 1")
    cancer_weight_list_2d = MLP_HELPER.get_mlp_weights(cancer_mlp_2d,cancer_train_output,0.001,4000)

    ################ Cancer MLP ##################
    mlp_class_results(cancer_test_output, cancer_weight_list_0,cancer_mlp_0,cancer_classes,"Cancer 0")
    # mlp_class_results(cancer_test_output, cancer_weight_list_1a,cancer_mlp_1a,cancer_classes,"Cancer 1a")
    mlp_class_results(cancer_test_output, cancer_weight_list_2b,cancer_mlp_2b,cancer_classes,"Cancer 2b")
    mlp_class_results(cancer_test_output, cancer_weight_list_2d,cancer_mlp_2d,cancer_classes,"Cancer 2d")

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
    de_class_results(cancer_test_output,cancer_mlp_0, de_popu_0, num_generations_de, cancer_classes, "Cancer 0")
    # for 1 layer
    de_class_results(cancer_test_output,cancer_mlp_1, de_popu_1, num_generations_de, cancer_classes, "Cancer 1")
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
    pso_class_results(cancer_test_output, cancer_mlp_0, all_popu_0, cancer_classes, "Cancer 0", p_size, num_particles, maxiter, shape1=all_popu_0[0][0].shape)
    # for 1 layer
    pso_class_results(cancer_test_output, cancer_mlp_1, all_popu_1, cancer_classes, "Cancer 1", p_size, num_particles, maxiter, shape1=all_popu_1[0][0].shape, shape2=all_popu_1[0][1].shape)
    # for 2 layers
    pso_class_results(cancer_test_output, cancer_mlp_2, all_popu_2, cancer_classes, "Cancer 2", p_size, num_particles, maxiter, shape1=all_popu_2[0][0].shape, shape2=all_popu_2[0][1].shape, shape3=all_popu_2[0][2].shape)

# print every machine outputs
def print_machine_set():
    print("################# Processing Regression Machine Dataset #######################")
    p_size = 200
    machine_mlp_0 = MLP(8, [], 1)
    machine_mlp_1 = MLP(8, [6], 1)
    machine_mlp_2 = MLP(8, [8,6], 1)

    # returns test datasets
    # machine_train_output, machine_test_output, x_max, x_min = MLP_HELPER.mlp_machine_data()
    machine_train_output, machine_test_output = MLP_HELPER.mlp_machine_data()
    machine_weight_list_0 = MLP_HELPER.get_mlp_weights(machine_mlp_0,machine_train_output,0.01,250)
    machine_weight_list_1 = MLP_HELPER.get_mlp_weights(machine_mlp_1,machine_train_output,0.01,500)
    machine_weight_list_2 = MLP_HELPER.get_mlp_weights(machine_mlp_2,machine_train_output,0.01,500)

    # ################ Machine MLP #################
    mlp_regress_results(machine_test_output, machine_weight_list_0,machine_mlp_0, x_min=None, x_max=None,df_name="Machine 0")
    mlp_regress_results(machine_test_output, machine_weight_list_1,machine_mlp_1, x_min=None, x_max=None,df_name="Machine 1")
    mlp_regress_results(machine_test_output, machine_weight_list_2,machine_mlp_2, x_min=None, x_max=None,df_name="Machine 2")

    # ################ Machine GA ###################
    # create machine GA population
    p_size = 200
    ga_popu_0 = create_population(p_size,machine_weight_list_0)
    ga_popu_1 = create_population(p_size,machine_weight_list_1)
    ga_popu_2 = create_population(p_size,machine_weight_list_2)

    # for 0 layers
    ga_regress_results(machine_test_output,machine_mlp_0, ga_popu_0, num_gens=200, t_size=2, c_prob=0.9, n_size=math.ceil(p_size*(2/3)), df_name="Machine 0")
    # for 1 layers
    ga_regress_results(machine_test_output,machine_mlp_1, ga_popu_1, num_gens=200, t_size=2, c_prob=0.9,  n_size=math.ceil(p_size*(2/3)), df_name="Machine 1")
    # for 2 layers
    ga_regress_results(machine_test_output,machine_mlp_2, ga_popu_2, num_gens=200, t_size=2, c_prob=0.9,  n_size=math.ceil(p_size*(2/3)), df_name="Machine 2")

    # ################ Machine DE ####################
    # create machine DE population
    p_size = 200
    de_popu_0 = create_population(p_size,machine_weight_list_0)
    de_popu_1 = create_population(p_size,machine_weight_list_1)
    de_popu_2 = create_population(p_size,machine_weight_list_2)
    num_generations_de = 200

    # for 0 layers
    de_regress_results(machine_test_output, machine_mlp_0, de_popu_0, num_generations_de,x_min=None, x_max=None,df_name="Machine 0",scale_factor=0.5)
    # for 1 layer
    de_regress_results(machine_test_output, machine_mlp_1, de_popu_1, num_generations_de,x_min=None, x_max=None,df_name="Machine 1",scale_factor=0.5)
    # for 2 layers
    de_regress_results(machine_test_output, machine_mlp_2, de_popu_2, num_generations_de,x_min=None, x_max=None,df_name="Machine 2",scale_factor=0.5)

    # ################ Machine PSO ####################
    p_size = 20
    all_popu_0 = create_population(p_size,machine_weight_list_0)
    all_popu_1 = create_population(p_size,machine_weight_list_1)
    all_popu_2 = create_population(p_size,machine_weight_list_2)
    num_particles = 10 
    maxiter = 30
    
    # for 0 layers
    pso_regress_results(machine_test_output, machine_mlp_0, all_popu_0, "Machine 0", p_size, num_particles, maxiter, shape1=all_popu_0[0][0].shape)
    # for 1 layer
    pso_regress_results(machine_test_output, machine_mlp_1, all_popu_1, "Machine 1", p_size, num_particles, maxiter, shape1=all_popu_1[0][0].shape, shape2=all_popu_1[0][1].shape)
    # for 2 layers
    pso_regress_results(machine_test_output, machine_mlp_2, all_popu_2, "Machine 2", p_size, num_particles, maxiter, shape1=all_popu_2[0][0].shape, shape2=all_popu_2[0][1].shape, shape3=all_popu_2[0][2].shape)
    
# print every forestfire outputs
def print_forestfires_set():
    print("################# Processing Regression Forestfire Dataset #######################")
    p_size = 200
    forestfires_mlp_0 = MLP(13, [], 1)
    forestfires_mlp_1 = MLP(13, [12], 1)
    forestfires_mlp_2 = MLP(13, [12,12], 1)

    # returns test datasets
    forestfires_train_output, forestfires_test_output = MLP_HELPER.mlp_forestfires_data()
    forestfires_weight_list_0 = MLP_HELPER.get_mlp_weights(forestfires_mlp_0,forestfires_train_output,0.01,250)
    forestfires_weight_list_1 = MLP_HELPER.get_mlp_weights(forestfires_mlp_1,forestfires_train_output,0.01,500)
    forestfires_weight_list_2 = MLP_HELPER.get_mlp_weights(forestfires_mlp_2,forestfires_train_output,0.01,500)

    # ################ Forest Fire MLP #################
    mlp_regress_results(forestfires_test_output, forestfires_weight_list_0,forestfires_mlp_0, x_min=None, x_max=None,df_name="Forestfires 0", prevent=True)
    mlp_regress_results(forestfires_test_output, forestfires_weight_list_1,forestfires_mlp_1, x_min=None, x_max=None,df_name="Forestfires 1", prevent=True)
    mlp_regress_results(forestfires_test_output, forestfires_weight_list_2,forestfires_mlp_2, x_min=None, x_max=None,df_name="Forestfires 2", prevent=True)

    # ################ Forest Fire GA ###################
    # create forestfires GA population
    p_size = 200
    ga_popu_0 = create_population(p_size,forestfires_weight_list_0)
    ga_popu_1 = create_population(p_size,forestfires_weight_list_1)
    ga_popu_2 = create_population(p_size,forestfires_weight_list_2)

    # for 0 layers
    ga_regress_results(forestfires_test_output,forestfires_mlp_0, ga_popu_0, num_gens=200, t_size=2, c_prob=0.9, n_size=math.ceil(p_size*(2/3)), x_min=None, x_max=None,df_name="Forestfires 0", prevent=True)
    # for 1 layers
    ga_regress_results(forestfires_test_output,forestfires_mlp_1, ga_popu_1, num_gens=200, t_size=2, c_prob=0.9, n_size=math.ceil(p_size*(2/3)), x_min=None, x_max=None,df_name="Forestfires 1", prevent=True)
    # for 2 layers
    ga_regress_results(forestfires_test_output,forestfires_mlp_2, ga_popu_2, num_gens=200, t_size=2, c_prob=0.9, n_size=math.ceil(p_size*(2/3)), x_min=None, x_max=None,df_name="Forestfires 2", prevent=True)

    # ################ Forest Fire DE ####################
    p_size = 200
    de_popu_0 = create_population(p_size,forestfires_weight_list_0)
    de_popu_1 = create_population(p_size,forestfires_weight_list_1)
    de_popu_2 = create_population(p_size,forestfires_weight_list_2)
    num_generations_de = 200

    # for 0 layers
    de_regress_results(forestfires_test_output, forestfires_mlp_0, de_popu_0, num_generations_de,x_min=None, x_max=None,df_name="Forestfires 0",scale_factor=0.5, prevent=True)
    # for 1 layer
    de_regress_results(forestfires_test_output, forestfires_mlp_1, de_popu_1, num_generations_de,x_min=None, x_max=None,df_name="Forestfires 1",scale_factor=0.5, prevent=True)
    # for 2 layers
    de_regress_results(forestfires_test_output, forestfires_mlp_2, de_popu_2, num_generations_de,x_min=None, x_max=None,df_name="Forestfires 2",scale_factor=0.5, prevent=True)

    # ################ Forest Fire PSO ####################
    p_size = 20
    all_popu_0 = create_population(p_size,forestfires_weight_list_0)
    all_popu_1 = create_population(p_size,forestfires_weight_list_1)
    all_popu_2 = create_population(p_size,forestfires_weight_list_2)
    num_particles = 10 
    maxiter = 30
    
    # for 0 layers
    pso_regress_results(forestfires_test_output, forestfires_mlp_0, all_popu_0, "Forestfires 0", p_size, num_particles, maxiter, shape1=all_popu_0[0][0].shape, prevent=True)
    # for 1 layer
    pso_regress_results(forestfires_test_output, forestfires_mlp_1, all_popu_1, "Forestfires 1", p_size, num_particles, maxiter, shape1=all_popu_1[0][0].shape, shape2=all_popu_1[0][1].shape, prevent=True)
    # for 2 layers
    pso_regress_results(forestfires_test_output, forestfires_mlp_2, all_popu_2, "Forestfires 2", p_size, num_particles, maxiter, shape1=all_popu_2[0][0].shape, shape2=all_popu_2[0][1].shape, shape3=all_popu_2[0][2].shape, prevent=True)

# print every abalone outputs
def print_abalone_set():
    print("################# Processing Regression Abalone Dataset #######################")
    p_size = 200
    abalone_mlp_0 = MLP(10, [], 1)
    abalone_mlp_1 = MLP(10, [6], 1)
    abalone_mlp_2 = MLP(10, [8,6], 2)

    # returns test datasets
    abalone_train_output, abalone_test_output = MLP_HELPER.mlp_abalone_data()
    abalone_weight_list_0 = MLP_HELPER.get_mlp_weights(abalone_mlp_0,abalone_train_output,0.01,250)
    abalone_weight_list_1 = MLP_HELPER.get_mlp_weights(abalone_mlp_1,abalone_train_output,0.01,500)
    abalone_weight_list_2 = MLP_HELPER.get_mlp_weights(abalone_mlp_2,abalone_train_output,0.01,500)

    # ################ Abalone MLP #################
    mlp_regress_results(abalone_test_output, abalone_weight_list_0,abalone_mlp_0, x_min=None, x_max=None,df_name="Abalone 0")
    mlp_regress_results(abalone_test_output, abalone_weight_list_1,abalone_mlp_1, x_min=None, x_max=None,df_name="Abalone 1")
    mlp_regress_results(abalone_test_output, abalone_weight_list_2,abalone_mlp_2, x_min=None, x_max=None,df_name="Abalone 2")

    # ################ Abalone GA ###################
    p_size = 200
    ga_popu_0 = create_population(p_size,abalone_weight_list_0)
    ga_popu_1 = create_population(p_size,abalone_weight_list_1)
    ga_popu_2 = create_population(p_size,abalone_weight_list_2)

    # for 0 layers
    ga_regress_results(abalone_test_output,abalone_mlp_0, ga_popu_0, num_gens=200, t_size=2, c_prob=0.9, n_size=math.ceil(p_size*(2/3)), x_min=None, x_max=None,df_name="Abalone 0")
    # for 1 layers
    ga_regress_results(abalone_test_output,abalone_mlp_1, ga_popu_1, num_gens=200, t_size=2, c_prob=0.9, n_size=math.ceil(p_size*(2/3)), x_min=None, x_max=None,df_name="Abalone 1")
    # for 2 layers
    ga_regress_results(abalone_test_output,abalone_mlp_2, ga_popu_2, num_gens=200, t_size=2, c_prob=0.9, n_size=math.ceil(p_size*(2/3)), x_min=None, x_max=None,df_name="Abalone 2")

    # ################ Abalone DE ####################
    p_size = 200
    de_popu_0 = create_population(p_size,abalone_weight_list_0)
    de_popu_1 = create_population(p_size,abalone_weight_list_1)
    de_popu_2 = create_population(p_size,abalone_weight_list_2)
    num_generations_de = 200

    # for 0 layers
    de_regress_results(abalone_test_output, abalone_mlp_0, de_popu_0, num_generations_de,x_min=None, x_max=None,df_name="Abalone 0",scale_factor=0.5)
    # for 1 layer
    de_regress_results(abalone_test_output, abalone_mlp_1, de_popu_1, num_generations_de,x_min=None, x_max=None,df_name="Abalone 1",scale_factor=0.5)
    # for 2 layers
    de_regress_results(abalone_test_output, abalone_mlp_2, de_popu_2, num_generations_de,x_min=None, x_max=None,df_name="Abalone 2",scale_factor=0.5)

    # ################ Abalone PSO ####################
    p_size = 20
    all_popu_0 = create_population(p_size,abalone_weight_list_0)
    all_popu_1 = create_population(p_size,abalone_weight_list_1)
    all_popu_2 = create_population(p_size,abalone_weight_list_2)
    num_particles = 10 
    maxiter = 30
    
    # for 0 layers
    pso_regress_results(abalone_test_output, abalone_mlp_0, all_popu_0, "Abalone 0", p_size, num_particles, maxiter, shape1=all_popu_0[0][0].shape)
    # for 1 layer
    pso_regress_results(abalone_test_output, abalone_mlp_1, all_popu_1, "Abalone 1", p_size, num_particles, maxiter, shape1=all_popu_1[0][0].shape, shape2=all_popu_1[0][1].shape)
    # for 2 layers
    pso_regress_results(abalone_test_output, abalone_mlp_2, all_popu_2, "Abalone 2", p_size, num_particles, maxiter, shape1=all_popu_2[0][0].shape, shape2=all_popu_2[0][1].shape, shape3=all_popu_2[0][2].shape)

if __name__ == "__main__":
    ############  Classification Instances  #####################################
    # print_glass_set()
    # print_soy_set()
    # print_cancer_set()

    # # ################  Regression Instances  #####################################
    # print_machine_set()
    # print_forestfires_set()
    print_abalone_set()

    # video_prints()