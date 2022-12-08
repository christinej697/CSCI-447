##########################################################################
# Xuying Swift and Christine Johnson, 11/2022, CSCI 447 Machine Learning
# Implements a Differential Evolution Algorithm
##########################################################################

import numpy as np
import random
import copy

from utils import UTILS
from mlp import MLP

class DE:
    def __init__(self, version: str, mlp: MLP, population, num_generations: int = 3, scale_factor = 1, crossover_probability: float = 0.9, classes = None, test_values = None, x_max = None, x_min = None,  verbose: str = None):
        self.version = version
        self.population = copy.deepcopy(population)
        self.ns = len(population)
        self.num_generations = num_generations
        self.beta  = scale_factor
        self.pr = crossover_probability
        self.classes = classes
        self.verbose = verbose
        self.fitness_dict = {}
        self.fit_keys = []
        self.population_loss = {}
        self.parents = []
        self.next_gen = []
        self.mlp = mlp
        self.test_values = test_values
        self.x_max = x_max
        self.x_min = x_min

    # calls all methods to run a full instance of DE for given number of generations
    def run(self):
        for i in range(self.num_generations):
            # new generation calculation
            self.pop_fitness()
            if self.verbose == True:
                print(f"Round {i+1} Performance",self.fitness_dict[self.fit_keys[-len(self.fit_keys)]])
            for xj in range(self.ns):
                # perform mutation on xj
                uj = self.mutation(copy.deepcopy(xj))
                # get crossover for xj and uj
                offspring = self.crossover(copy.deepcopy(self.population[xj]), uj)
                # choose who will go in next generation
                fit_offspring = self.one_fitness(offspring)
                if self.version == "class":
                    if fit_offspring > self.fitness_dict[xj]:
                        self.next_gen.append(copy.deepcopy(offspring))
                    else:
                        self.next_gen.append(copy.deepcopy(self.population[xj]))
                elif self.version == "regress":
                    if fit_offspring < self.fitness_dict[xj]:
                        self.next_gen.append(copy.deepcopy(offspring))
                    else:
                        self.next_gen.append(copy.deepcopy(self.population[xj]))
            # generational replace population
            self.population = copy.deepcopy(self.next_gen)
            # self.n_replacement
            self.next_gen = []
        self.pop_fitness
        if self.verbose == True:
            print("ALL DONE")
            print(f"Final Round {i+1} Performance",self.fitness_dict[self.fit_keys[-len(self.fit_keys)]])
        return self.population_loss[self.fit_keys[-len(self.fit_keys)]] #self.fitness_dict[self.fit_keys[-len(self.fit_keys)]]

    # get fitness ranking of population
    def pop_fitness(self):
        # we trying to max F1 score fore each set of weights
        # y = F1 score for each chrome
        # we need to loop throuhg the population and then calculate its F1 scores
        for i in range(len(self.population)):
            # result = UTILS.get_performance(UTILS, self.population[i], self.classes)
            result = UTILS.get_performance(UTILS, self.mlp, self.population[i], self.classes, self.test_values)
            if self.version == "class":
                loss = UTILS.calculate_loss_np(UTILS, result, self.classes)
                self.fitness_dict[i] = loss["Accuracy"]
                self.population_loss[i] = loss
            elif self.version == "regress":
                loss = UTILS.calculate_loss_for_regression(UTILS, result, self.test_values, self.x_max, self.x_min)
                # print(loss)
                self.fitness_dict[i] = loss["MSE"]
                self.population_loss[i] = loss
            # self.fitness_dict[i] = loss["F1"]
        if self.version == "class":
            sorted_by_f1 = sorted(self.fitness_dict.items(), key=lambda x:x[1], reverse=True)
        elif self.version == "regress":
            sorted_by_f1 = sorted(self.fitness_dict.items(), key=lambda x:x[1], reverse=False)
        converted_dict = dict(sorted_by_f1)
        self.fitness_dict = copy.deepcopy(converted_dict)
        # if self.verbose == True:
            # print("Fitness ranking")
            # print(self.fitness_dict)
            # print("\n-----------------------------------------\n")
        self.fit_keys = list(self.fitness_dict.keys())
        # if self.verbose == True:
            # print(self.fitness_dict)
            # print(self.fit_keys)
            # print(self.fit_keys)

    # return the performance of an individual participant
    def one_fitness(self,x):
        result = UTILS.get_performance(UTILS, self.mlp, x, self.classes, self.test_values)
        if self.version == "class":
            loss = UTILS.calculate_loss_np(UTILS, result, self.classes)
            return loss["Accuracy"]
        elif self.version == "regress":
            loss = UTILS.calculate_loss_for_regression(UTILS, result, self.test_values, self.x_max, self.x_min)
            # print(loss)
            return loss["MSE"]

        
    # create trial vector by applying the mutation operator
    # target vector is selected at random
    def mutation(self, index):
        # select k vectors other from the population
        candidates = copy.deepcopy(self.population)
        candidates.pop(index)
        k_vectors = random.choices(candidates, k=3)
        # print("k_vector",k_vectors[1])
        # calculate the trial vector
        uj = np.subtract(k_vectors[1], k_vectors[2])
        uj = uj * self.beta
        uj = np.add(k_vectors[0], uj)
        if self.verbose == True:
            print("uj:",uj)
        return uj

    # create an offspring using binomial crossover
    def crossover(self, xj, uj):
        if random.random() < self.pr:
            offspring = copy.deepcopy(xj)
            # perform uniform crossover on all genes
            for (item1, item2) in zip(offspring, uj):
                # print(np.array(item1).shape)
                # print(np.array(item2).shape)
                for row_idx in range(len(item1)):
                    for col_idx in range(len(item1[row_idx])):
                        # decide if swap values for the gene
                        # print(row_idx,",",col_idx)
                        if 0.5 > random.random():
                            item1[row_idx][col_idx] == item2[row_idx][col_idx]
        else:
            offspring = copy.deepcopy(uj)

        return offspring

    # generational replacement, replace all n old generation with all n children
    def n_replacement(self):
        for i in range(100):
            self.population[self.fit_keys[-i-1]] = copy.deepcopy(self.next_gen[i])
