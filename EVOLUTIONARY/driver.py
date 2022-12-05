from mlp import MLP
from utils import UTILS
import pandas as pd
from random import random
import json
import numpy as np
from genetic_alg import GA
from diff_evolution import DE
import sys
import math
from pso_alg import PSO
from mlp_helper import MLP_HELPER

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
            all_popu.append(idv)

    return population

if __name__ == "__main__":
    size = 20
    all_popu = []
    glass_classes = 
    print("################# Processing Classification Glass Dataset #######################")
    glass_mlp = MLP(10, [6], 7)
    glass_test_output = MLP_HELPER.mlp_glass_data()
    glass_weight_list = MLP_HELPER.get_mlp_weights(glass_mlp, glass_test_output,0.01,1000)
    # print(type(glass_weight_list))
    # print(type(glass_weight_list[-1]))
    # for item in glass_weight_list[-1]:
    #     print("ITEM:",item)
    #     print("1:",item[1])
    # autopopulate weight population to p_size
    all_popu = create_population(size,glass_weight_list)
    ga = GA("class",all_popu,num_generations=100,tournament_size=2, crossover_probability=0.9,classes = glass_classes)
    # print("################# Processing Classification Cancer Dataset #######################")
    # cancer_mlp = MLP(10, [6], 2)
    # cancer_test_output = MLP_HELPER.mlp_cancer_data()
    # cancer_weight_list = MLP_HELPER.get_mlp_weights(cancer_mlp, cancer_test_output,0.01,1000)

    # print("################# Processing Classification Soybean Dataset #######################")
    # soybean_mlp = MLP(22, [6], 4)
    # soybean_test_output = MLP_HELPER.mlp_soybean_data()
    # soybean_weight_list = MLP_HELPER.get_mlp_weights(soybean_mlp , soybean_test_output,0.01,1000)

    # print("################# Processing Regression Machine Dataset #########################")
    # machine_mlp = MLP(8, [6], 1)
    # machine_test_output = MLP_HELPER.mlp_machine_data()
    # machine_weight_list = MLP_HELPER.get_mlp_weights(machine_mlp, machine_test_output,0.01,1000)

    # print("################# Processing Regression Abalone Dataset #########################")
    # abalone_mlp = MLP(10, [6], 1)
    # abalone_test_output = MLP_HELPER.mlp_abalone_data()
    # abalone_weight_list = MLP_HELPER.get_mlp_weights(abalone_mlp, abalone_test_output,0.01,1000)

    # print("################# Processing Regression Forestfires Dataset #########################")
    # forestfires_mlp = MLP(13, [6], 1)
    # forestfires_test_output = MLP_HELPER.mlp_forestfires_data()
    # forestfires_weight_list = MLP_HELPER.get_mlp_weights(forestfires_mlp, forestfires_test_output,0.01,1000)

    # print(cancer_weight_list,"\n\n")
    # print(cancer_weight_list[-2],"\n")
    # print(cancer_weight_list[-1])


    # learning_rate = 0.01

    # iterations = 1000    
    # ###############glass dataset ############################
    
    # # glass_mlp = MLP(10, [], 7)
    # # mlp_glass_data(glass_mlp, learning_rate, iterations)
   
    # glass_mlp = MLP(10, [6,6], 7)
    # target_output_dict, best_num, populations= mlp_glass_data(glass_mlp, learning_rate, iterations)
    # classes = [1, 2, 3, 4, 5, 6, 7]
    # best_weights = target_output_dict[best_num]
    # # print("best weights population: ", best_weights)
    # version = "classification" 
    # population = best_weights
    # num_generations = 10
    # tournament_size = 2
    # crossover_probability = 0.9
    # # print("populations")
    # # print(population)

    # ################
    # # initialize pop
    # # how to create a population with size 10
    # size = 200
    # all_popu = []
    # p_size = population.shape

    # for i in range(size):
    #     new_p = np.random.uniform(-0.01, 0.01, p_size)
    #     all_popu.append(new_p)
    # all_popu.append(population)
    # # # ga = GA(version, all_popu, num_generations, tournament_size, crossover_probability, classes=classes)
    # # ga = GA(version, all_popu, num_generations, tournament_size, crossover_probability, classes=classes, n_size=math.ceil(size*(2/3)))
    # # ga.fitness()
    # # print(ga.fit_keys)
    # # print(ga.fitness_dict)
    # # print(len(ga.fit_keys))
    # # print("Round 0 Performance",ga.fitness_dict[ga.fit_keys[-len(ga.fit_keys)]],"\n")
    # # # ga.selection()
    # # # children = ga.crossover()
    # # ga.n_selection()
    # # children = ga.n_crossover()
    # # children = ga.mutation(children)
    # # # generational replacement, replace all n old generation with all n children
    # # # ga.replacement(children)
    # # ga.n_replacement(children)
    # # for i in range(200):
    # #     ga.fitness()
    # #     print(f"Round {i+1} Performance",ga.fitness_dict[ga.fit_keys[-len(ga.fit_keys)]]) 
    # #     # ga.selection()
    # #     # children = ga.crossover() 
    # #     ga.n_selection()
    # #     children = ga.n_crossover()
    # #     children = ga.mutation(children)
    # #     # generational replacement, replace all n old generation with all n children
    # #     ga.n_replacement(children)
    # #     # ga.replacement(children)
    # #     print()
    # # print("ALL DONE!")
    # # sys.exit(0)

    # # ga.fitness(classes)
    # # parents = ga.selection()

    # # print("all populations")
    # # print(all_popu)

    # ###### GA algorithm ##############

    # # initialize the population. And the population is a single fold test output from MLP network with the best formance

    # # evaluate all populations fitnesss, we ranck their fitness by best to worst

    # # random select k solutions from the population, and we use tournament method to select best solutions, and then use them as parents.

    # # crossover to create next generation, we use a crossover probility variable to do uniform crossover

    # # use a mutation probability to do mutation and add a tunable weight to each weight in our children

    # # we need to evaluate the fitness of the newly created children

    # # replacement using steady state selection, get rid of the k worse solutions and replace them with the newly generated children. 

    # # terminatin: a set number of generations or until performance is not improving anymore

    # # print("ITERATION is: ", iterations)
    # # glass_mlp = MLP(10, [5, 5], 7)
    # # mlp_glass_data(glass_mlp, learning_rate, iterations)

    # # print()
    # # print("---------------------------------------------------------------------------------")
    # # mlp_glass_data(glass_mlp, learning_rate, iterations)
    # # ################cancer dataset #######################
    # # cancer_mlp = MLP(10, [], 2)
    # # print()
    # # print("---------------------------------------------------------------------------------")
    # # mlp_cancer_data(cancer_mlp, learning_rate, iterations)

    # # # #################soybean dataset ######################
    # # print("---------------------------------------------------------------------------------")
    # # soybean_mlp = MLP(22, [12, 12], 4)
    # # mlp_soybean_data(soybean_mlp, learning_rate, iterations)

    # ########### DIFF EVOLUTION ######################
    # num_generations = 200
    # crossover_probability = 0.5
    # size = 200
    # all_popu = []
    # p_size = population.shape

    # for i in range(size):
    #     new_p = np.random.uniform(-0.01, 0.01, p_size)
    #     all_popu.append(new_p)
    # all_popu.append(population)


    # # de = DE(version, all_popu, num_generations,classes=classes)
    # # de.run()
    # # # de.fitness()
    # # # print(de.fit_keys)
    # # # print(de.fitness_dict)
    # # # print(len(de.fit_keys))
    # # # print("Round 0 Performance",de.fitness_dict[de.fit_keys[-len(de.fit_keys)]],"\n")
    # # # ga.selection()
    # # # children = ga.crossover()
    # # print("ALL DONE!")


    # ############ PSO ########################
    # print("Entering PSO")
    # w = 0.8
    # c1 = 0.1
    # c2 = 0.1
    # # Create particles
    # n_particles = 50
    # run_nums = 50
    # pso = PSO(all_popu, c1, c2, w, n_particles)
    # for i in range(50):
    #     pso.run()
