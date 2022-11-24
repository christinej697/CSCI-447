##########################################################################
# Xuying Swift and Christine Johnson, 11/2022, CSCI 447 Machine Learning
# Implements a Differential Evolution Algorithm
##########################################################################

import numpy as np
import random
from utils import UTILS

class DE:
    def __init__(self, version: str, population, num_generations: int = 3, scale_factor = 1, crossover_probability: float = 0.9
    , classes = None, verbose: str = None):
        self.version = version
        self.population = population
        self.ns = len(population)
        self.num_generations = num_generations
        self.beta  = scale_factor
        self.pr = crossover_probability
        self.classes = classes
        self.verbose = verbose
        self.fitness_dict = {}
        self.fit_keys = []
        self.parents = []

    def run(self):
        pass

    # get fitness ranking of population
    def fitness(self):
        # print("population")
        # print(self.population)
        # print("\n-----------------------------------------\n")
        # we trying to max F1 score fore each set of weights
        # y = F1 score for each chrome
        # we need to loop throuhg the population and then calculate its F1 scores
        for i in range(len(self.population)):
            result = UTILS.get_performance(UTILS, self.population[i], self.classes)
            loss = UTILS.calculate_loss_np(UTILS, result, self.classes)
            self.fitness_dict[i] = loss["F1"]
        sorted_by_f1 = sorted(self.fitness_dict.items(), key=lambda x:x[1], reverse=True)
        converted_dict = dict(sorted_by_f1)
        self.fitness_dict = converted_dict.copy()
        # print("Fitness ranking")
        # print(self.fitness_dict)
        # print("\n-----------------------------------------\n")
        self.fit_keys = list(self.fitness_dict.keys())
        # print(self.fit_keys)

    # create trial vector by applying the mutation operator
    # target vector is selected at random
    def mutation(self, xj):
        # select k vectors other from the population
        k_vectors = random.choices(self.population.copy().remove(xj), k=3)
        # calculate the trial vector
        uj = np.subtract(k_vectors[1] - k_vectors[2])
        uj = uj * self.beta
        uj = np.add(k_vectors[0], uj)

        return uj

    # create an offspring using binomial crossover
    def crossover(self, xj, uj):
        if random.random() < self.pr:
            offspring = xj.copy()
            # perform uniform crossover on all genes
            for row_idx in range(len(offspring)):
                for col_idx in range(len(offspring[row_idx])):
                    # decide if swap values for the gene
                    if 0.5 > random.random():
                        offspring[row_idx][col_idx] == uj[row_idx][col_idx]
        else:
            offspring = uj

        return offspring
    
    # if offspring has better performance than xj, replace xj with offspring
    def fitness(self, xj, offspring):
        
        pass

    # generational replacement, replace all n old generation with all n children
    def replacement(self, children):
        self.population = children

    # termination: a set number of generations or until performance is not improving anymore