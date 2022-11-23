##########################################################################
# Xuying Swift and Christine Johnson, 11/2022, CSCI 447 Machine Learning
# Implements a Differential Evolution Algorithm
##########################################################################

import numpy as np
import random
from utils import UTILS

class DE:
    def __init__(self, version: str, population, num_generations: int = 3, mutation_k: int = 3, scale_factor = 1, crossover_probability: float = 0.9
    , classes = None, verbose: str = None):
        self.version = version
        self.population = population
        self.pop_size = len(population)
        self.num_generations = num_generations
        self.k = mutation_k
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

    # create an offspring using binomial crossover
    def crossover(self):
        pass

    # create trial vector by applying the mutation operator
    def mutation(self, children):
        pass

    # generational replacement, replace all n old generation with all n children
    def replacement(self, children):
        self.population = children

    # termination: a set number of generations or until performance is not improving anymore