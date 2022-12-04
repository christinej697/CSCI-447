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

def mlp_machine_data(machine_mlp, learning_rate, iterations):
    machine_labels = ["vendor_name","model","myct","mmin","mmax","cach","chmin","chmax","prp","erp"]
    machine_df = UTILS.import_data("machine.data",machine_labels)
    machine_df = machine_df.drop(['vendor_name','model'], axis = 1)
    machine_df = UTILS.min_max_normalization(machine_df)
    machine_training1,machine_testing1,machine_training2,machine_testing2,machine_training3,machine_testing3,machine_training4,machine_testing4,machine_training5,machine_testing5,machine_training6,machine_testing6,machine_training7,machine_testing7,machine_training8,machine_testing8,machine_training9,machine_testing9,machine_training10,machine_testing10,machine_tuning = UTILS.stratify_and_fold_regression(machine_df)

    return machine_training1,machine_testing1,machine_training2,machine_testing2,machine_training3,machine_testing3,machine_training4,machine_testing4,machine_training5,machine_testing5,machine_training6,machine_testing6,machine_training7,machine_testing7,machine_training8,machine_testing8,machine_training9,machine_testing9,machine_training10,machine_testing10,machine_tuning

def mlp_glass_data(glass_mlp, learning_rate, iterations):
    # print(iterations)
    glass_labels = ['id-num','retractive-index','sodium','magnesium','aluminum','silicon','potasium','calcium','barium','iron','class']
    glass_df = UTILS.import_data(UTILS, "glass.data", glass_labels)
    glass_df.drop(columns=glass_df.columns[0], axis=1, inplace=True)
    # print()
    # print("glass dataframe: ")
    # print(glass_df)

    new_glass_df = UTILS.min_max_normalization(UTILS, glass_df)
    # print()
    # print("normalized glass dataframe: ")
    # print(new_glass_df)

    glass_training1,glass_testing1,glass_training2,glass_testing2,glass_training3,glass_testing3,glass_training4,glass_testing4,glass_training5,glass_testing5,glass_training6,glass_testing6,glass_training7,glass_testing7,glass_training8,glass_testing8,glass_training9,glass_testing9,glass_training10,glass_testing10,glass_tuning = UTILS.stratify_and_fold_classification(UTILS, new_glass_df)
    train_list = [glass_training1, glass_training2, glass_training3, glass_training4, glass_training5, glass_training6, glass_training7, glass_training8, glass_training9, glass_training10]
    test_list = [glass_testing1, glass_testing2, glass_testing3, glass_testing4, glass_testing5, glass_testing6, glass_testing7, glass_testing8, glass_testing9, glass_testing10]
    classes = [1, 2, 3, 4, 5, 6, 7]
    target_output_dict = {}
    print("2",glass_mlp.weights)
    performance, populations = data_processing(train_list, test_list, glass_mlp, learning_rate, iterations, classes, target_output_dict)
    # print(performance)
    # print(populations)
    # print()
    # print("target output dict : ",target_output_dict)
    # print()
    # print("Final resuld and performance for glass data: ")
    loss_dict, best_num = get_loss(performance, classes)
    # print("loss_dict,", loss_dict)
    # print("END")
    with open('glass_result.txt', 'w+') as convert_file:
     convert_file.write(json.dumps(loss_dict))
    return target_output_dict, best_num, populations, glass_training1,glass_testing1,glass_training2,glass_testing2,glass_training3,glass_testing3,glass_training4,glass_testing4,glass_training5,glass_testing5,glass_training6,glass_testing6,glass_training7,glass_testing7,glass_training8,glass_testing8,glass_training9,glass_testing9,glass_training10,glass_testing10,glass_tuning


def get_loss(performances, classes):
    loss_dict = {}
    loss_sum = 0
    couter = 1
    best_f1 = 0
    best_num = 1
    for i in performances:
        loss = UTILS.calculate_loss_function(UTILS, i, classes, "classification")
        if loss["F1"] > best_f1:
            best_f1 = loss["F1"]
            best_num = couter
        loss_sum += loss['F1']
        loss_dict[couter] = loss
        #print("test case number: {}, loss: {}".format(couter, loss))
        couter += 1
    #print(loss_sum)
    avg_p = loss_sum / couter
    # print()
    # print("The average F1 score of 10 folds is: ", avg_p)
    return loss_dict, best_num
        
def data_processing(train_list, test_list, mlp, learing_rate, iterations, classes,  target_output_dict):
    performance = []
    counter = 1
    populations = []
    for i in range(len(train_list)):
        training_np = train_list[i].to_numpy()
        # print()
        # print("training" ,i+1, "numpy: ")
        # print(training_np)

        training_targets_df = train_list[i]["class"]
        training_targets_np = training_targets_df.to_numpy()
        # print()
        # print("training", i+ 1, "_targets")
        # print(training_targets_np)

        #print("Taining~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        mlp.train_network(training_np, training_targets_np, iterations, learing_rate)
        #########################################################################
        testing_np = test_list[i].to_numpy()
        # print("Testing NP:",testing_np)
        # print()
        # print("testing", i+1, "_numpy: ")
        # print(testing_np)
        # # Train on our testsets
        #print("Testing~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        # print("Target ouputs:",testing_np)
        test_output = mlp.forward_feed(testing_np)
        # print("Testing Output:",test_output)
        # print("test_output")
        # print(test_output)
        target_output_dict[counter] = test_output

        result = mlp.find_max_value(test_output, classes)
        # print("!!!!!!!!!!!!!!!!!!!")
        # print(result)
        testing_targets = test_list[i]["class"].to_numpy()
    
        performance_df = pd.DataFrame(test_list[i]["class"])
        performance_df["prediction"] = result
        #print(performance_df)
        performance.append(performance_df)
        # populations[counter] = test_output
        populations.append(mlp.weights)
        counter += 1
        print("3",glass_mlp.weights)
    print("pop",populations,"end")
    print("pop",populations[-1],"end")
    print(glass_mlp.weights)
    return performance, populations
    


if __name__ == "__main__":
    learning_rate = 0.01
    iterations = 2
    ###############glass dataset ############################
    
    # glass_mlp = MLP(10, [], 7)
    # mlp_glass_data(glass_mlp, learning_rate, iterations)
   

    machine = MLP(9, [6,6], 1)

    glass_mlp = MLP(10, [6,6], 7)
    print("1",glass_mlp.weights)
    target_output_dict, best_num, populations, glass_training1,glass_testing1,glass_training2,glass_testing2,glass_training3,glass_testing3,glass_training4,glass_testing4,glass_training5,glass_testing5,glass_training6,glass_testing6,glass_training7,glass_testing7,glass_training8,glass_testing8,glass_training9,glass_testing9,glass_training10,glass_testing10,glass_tuning = mlp_glass_data(glass_mlp, learning_rate, iterations)
    classes = [1, 2, 3, 4, 5, 6, 7]
    best_weights = target_output_dict[best_num]
    # print("best weights population: ", best_weights)
    version = "classification" 
    population = best_weights
    num_generations = 10
    tournament_size = 2
    crossover_probability = 0.9
    # print("populations")
    # print(population)

    ################
    # initialize pop
    # how to create a population with size 10
    size = 20
    all_popu = []
  # print("pop",population,"end")
    # p_size = population.shape
    p_size = populations[0].shape

    for i in range(size):
        new_p = np.random.uniform(-0.01, 0.01, p_size)
        all_popu.append(new_p)
    # all_popu.append(population)
    all_popu.append(populations)
    # ga = GA(version, all_popu, num_generations, tournament_size, crossover_probability, classes=classes)
    ga = GA(glass_mlp, glass_training5, glass_testing5, version, all_popu, num_generations, tournament_size, crossover_probability, classes=classes, n_size=math.ceil(size*(2/3)))
    ga.fitness()
    sys.exit(0)
    print(ga.fit_keys)
    print(ga.fitness_dict)
    print(len(ga.fit_keys))
    print("Round 0 Performance",ga.fitness_dict[ga.fit_keys[-len(ga.fit_keys)]],"\n")
    # ga.selection()
    # children = ga.crossover()
    ga.n_selection()
    children = ga.n_crossover()
    children = ga.mutation(children)
    # generational replacement, replace all n old generation with all n children
    # ga.replacement(children)
    ga.n_replacement(children)
    for i in range(200):
        ga.fitness()
        print(f"Round {i+1} Performance",ga.fitness_dict[ga.fit_keys[-len(ga.fit_keys)]]) 
        # ga.selection()
        # children = ga.crossover() 
        ga.n_selection()
        children = ga.n_crossover()
        children = ga.mutation(children)
        # generational replacement, replace all n old generation with all n children
        ga.n_replacement(children)
        # ga.replacement(children)
        print()
    print("ALL DONE!")

    # ga.fitness(classes)
    # parents = ga.selection()

    # print("all populations")
    # print(all_popu)

    ###### GA algorithm ##############

    # initialize the population. And the population is a single fold test output from MLP network with the best formance

    # evaluate all populations fitnesss, we ranck their fitness by best to worst

    # random select k solutions from the population, and we use tournament method to select best solutions, and then use them as parents.

    # crossover to create next generation, we use a crossover probility variable to do uniform crossover

    # use a mutation probability to do mutation and add a tunable weight to each weight in our children

    # we need to evaluate the fitness of the newly created children

    # replacement using steady state selection, get rid of the k worse solutions and replace them with the newly generated children. 

    # terminatin: a set number of generations or until performance is not improving anymore

    # print("ITERATION is: ", iterations)
    # glass_mlp = MLP(10, [5, 5], 7)
    # mlp_glass_data(glass_mlp, learning_rate, iterations)

    # print()
    # print("---------------------------------------------------------------------------------")
    # mlp_glass_data(glass_mlp, learning_rate, iterations)
    # ################cancer dataset #######################
    # cancer_mlp = MLP(10, [], 2)
    # print()
    # print("---------------------------------------------------------------------------------")
    # mlp_cancer_data(cancer_mlp, learning_rate, iterations)

    # # #################soybean dataset ######################
    # print("---------------------------------------------------------------------------------")
    # soybean_mlp = MLP(22, [12, 12], 4)
    # mlp_soybean_data(soybean_mlp, learning_rate, iterations)

    ########### DIFF EVOLUTION ######################
    num_generations = 200
    crossover_probability = 0.5
    size = 200
    all_popu = []
    p_size = population.shape

    for i in range(size):
        new_p = np.random.uniform(-0.01, 0.01, p_size)
        all_popu.append(new_p)
    all_popu.append(population)
    de = DE(version, all_popu, num_generations,scale_factor = 0.5, crossover_probability=crossover_probability, classes=classes)
    de.run()
    # de.fitness()
    # print(de.fit_keys)
    # print(de.fitness_dict)
    # print(len(de.fit_keys))
    # print("Round 0 Performance",de.fitness_dict[de.fit_keys[-len(de.fit_keys)]],"\n")
    # ga.selection()
    # children = ga.crossover()
    print("ALL DONE!")
    sys.exit(0)