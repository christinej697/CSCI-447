import numpy as np
from utils import UTILS
import pandas as pd

class GA:
    def __init__(self, version, population, num_generations, tournament_size, crossover_probability):
        self.version = version
        self.population = population
        self.num_generations = num_generations
        self.tournament_size = tournament_size
        self.pr = crossover_probability

    # method that implements tournament selection
    def selection(self):
        pass

    def crossover(self):
        pass

    def fitness(self, classes):
        fitness_dict = {}
        # we trying to max F1 score fore each set of weights
        # y = F1 score for each chrome
        # we need to loop throuhg the population and then calculate its F1 scores
        for i in range(len(self.population)):
            result = UTILS.get_performance(UTILS, self.population[i], classes)
            loss = UTILS.calculate_loss_np(UTILS, result, classes)
            fitness_dict[i] = loss["F1"]
        sorted_by_f1 = sorted(fitness_dict.items(), key=lambda x:x[1], reverse=True)
        converted_dict = dict(sorted_by_f1)
        print(converted_dict)

        

    # initialize the population. And the population is a single fold test output from MLP network with the best formance

    # evaluate all populations fitnesss, we ranck their fitness by best to worst

    # random select k solutions from the population, and we use tournament method to select best solutions, and then use them as parents.

    # crossover to create next generation, we use a crossover probility variable to do uniform crossover

    # use a mutation probability to do mutation and add a tunable weight to each weight in our children

    # we need to evaluate the fitness of the newly created children

    # replacement using steady state selection, get rid of the k worse solutions and replace them with the newly generated children. 

    # terminatin: a set number of generations or until performance is not improving anymore