##########################################################################
# Xuying Swift and Christine Johnson, 11/2022, CSCI 447 Machine Learning
# Implements a Genetic Algorithm with a steady state population, tournament selection,
# uniform crossover, and generational replacement
##########################################################################

import sys
import copy
import numpy as np
import random

from utils import UTILS
from mlp import MLP

class GA:
    def __init__(self, version: str, mlp: MLP, population, num_generations: int = 3, tournament_size: int = 3, crossover_probability: float = 0.9, coin_toss: float = 0.5, mutation_probability: float = 0.02, mutate_sigma: float = 0.1, classes = None, n_size= None, test_values = None, x_max = None, x_min = None, verbose: str = None):
        self.version = version
        self.population = copy.deepcopy(population)
        self.pop_size = len(population)
        self.num_generations = num_generations
        self.tournament_size = tournament_size
        self.pr = crossover_probability
        self.coin_toss = coin_toss
        self.pm = mutation_probability
        self.mutate_sigma = mutate_sigma
        self.classes = classes
        self.verbose = verbose
        self.fitness_dict = {}
        self.fit_keys = []
        self.population_loss = {}
        self.parents = []
        self.n_size = n_size
        self.mlp = mlp
        self.test_values = test_values
        self.x_max = x_max
        self.x_min = x_min

    # calls all methods to run a full instance of GA for given number of generations
    def run(self):
        for i in range(self.num_generations):
            self.fitness()
            if self.verbose == True:
                print(f"Round {i} Performance: {self.fitness_dict[self.fit_keys[-len(self.fit_keys)]]}")
                # print(self.fitness_dict)
                # print(self.fit_keys)
            self.selection()
            self.n_selection()
            # children = self.crossover()
            children = self.n_crossover()
            children = self.mutation(children)
            # generational replacement, replace all n old generation with all n children
            # self.population = children
            self.n_replacement(children)
        return self.population_loss[self.fit_keys[-len(self.fit_keys)]] #self.fitness_dict[self.fit_keys[-len(self.fit_keys)]]

    # get fitness ranking of population
    def fitness(self):
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


    # use tournament method: select k random participants from the population, then the best of the k
    # use above to produce and return all parents for generational replacement
    def selection(self):
        self.parents = []
        while len(self.parents) != self.pop_size:
            # randomly select k participants from the population
            # tournament_pool = random.choices(self.population, k=self.tournament_size)
            # fit_keys = list(self.fitness_dict.keys())
            tournament_pool = random.choices(list(range(0,self.pop_size)), k=self.tournament_size)
            pool_best = tournament_pool[0]
            if self.verbose == True:
                print("Tournament pool:",tournament_pool)
            # select pool participant with the best fitness as parent
            for p in tournament_pool:
                # if self.verbose == True:
                    # print("best",pool_best,":",self.fitness_dict[pool_best]," , ","current",p,":",self.fitness_dict[p])
                # if p fitness > pool_best fitness:
                if self.version == "class":
                    if self.fitness_dict[p] > self.fitness_dict[pool_best]:
                        # if self.verbose == True:
                            # print("New Best",p,"!")
                        pool_best = p
                elif self.version == "regress":
                    if self.fitness_dict[p] < self.fitness_dict[pool_best]:
                        # if self.verbose == True:
                            # print("New Best",p,"!")
                        pool_best = p
            self.parents.append(copy.deepcopy(self.population[pool_best]))
            # if self.verbose == True:
                # print()
            #     print("parents",self.parents)
        # if self.verbose == True:
            # print("Final parents",self.parents)

    # use tournament method: select k random participants from the population, then the best of the k
    # use above to produce and return all parents for n replacement
    def n_selection(self):
        self.parents = []
        while len(self.parents) != self.n_size:
            # randomly select k participants from the population
            # tournament_pool = random.choices(self.population, k=self.tournament_size)
            # fit_keys = list(self.fitness_dict.keys())
            tournament_pool = random.choices(list(range(0,self.pop_size)), k=self.tournament_size)
            pool_best = tournament_pool[0]
            # if self.verbose == True:
                # print("Tournament pool:",tournament_pool)
            # select pool participant with the best fitness as parent
            for p in tournament_pool:
                # if self.verbose == True:
                    # print("best",pool_best,":",self.fitness_dict[pool_best]," , ","current",p,":",self.fitness_dict[p])
                # if p fitness > pool_best fitness:
                if self.version == "class":
                    if self.fitness_dict[p] > self.fitness_dict[pool_best]:
                        # if self.verbose == True:
                            # print("New Best",p,"!")
                        pool_best = p
                elif self.version == "regress":
                    if self.fitness_dict[p] < self.fitness_dict[pool_best]:
                        # if self.verbose == True:
                            # print("New Best",p,"!")
                        pool_best = p
            self.parents.append(copy.deepcopy(self.population[pool_best]))
            # if self.verbose == True:
                # print()
            #     print("parents",self.parents)
        # if self.verbose == True:
            # print("Final parents",self.parents)

    # use uniform crossover to produce next generation from parents, w/ crossover probability
    def crossover(self):
        children = []
        index = 0
        while len(children) != self.pop_size:
            # if self.verbose == True:
                # print(f"Crossover now has {len(children)} children, need {self.pop_size}")
            # check if we will perform crossover acccording to crossover probability
            if random.random() < self.pr:
                # if self.verbose == True:
                    # print(f"Performing Crossover")
                # if only one parent left, use parent as one replacement child
                if index >= self.pop_size - 1:
                    parent_1 = self.parents[index]
                    children.append(copy.deepcopy(parent_1))
                    index += 1
                # for two parents, product two replacement children
                else:
                    # select two parents
                    parent_1 = self.parents[index]
                    parent_2 = self.parents[index + 1]
                    child_1 = copy.deepcopy(parent_1)
                    child_2 = copy.deepcopy(parent_2)
                    # perform uniform crossover on all genes
                    for row_idx in range(len(parent_1)):
                        for col_idx in range(len(parent_1[row_idx])):
                            # decide if swap values for the gene
                            if self.coin_toss > random.random():
                                child_1[row_idx][col_idx] == child_2[row_idx][col_idx]
                                child_2[row_idx][col_idx] == child_1[row_idx][col_idx]
                    children.extend([child_1, child_2])
                    index += 2
            # according to crossover probability use parent instead of crossover
            else:
                # if only one parent left, use parent as one replacement child
                if index >= self.pop_size - 1:
                    parent_1 = self.parents[index]
                    children.append(copy.deepcopy(parent_1))
                    index += 1
                else:
                    # select two parents
                    parent_1 = self.parents[index]
                    parent_2 = self.parents[index + 1]
                    # if self.verbose == True:
                        # print(f"Don't Perform Crossover")
                    children.extend([copy.deepcopy(parent_1), copy.deepcopy(parent_2)])
                    index += 2

        return children

    # use uniform crossover to produce next generation  of size n from parents, w/ crossover probability
    def n_crossover(self):
        children = []
        index = 0
        while len(children) != self.n_size:
            # if self.verbose == True:
                # print(f"Crossover now has {len(children)} children, need {self.pop_size}")
            # check if we will perform crossover acccording to crossover probability
            if random.random() < self.pr:
                # if self.verbose == True:
                    # print(f"Performing Crossover")
                # if only one parent left, use parent as one replacement child
                if index >= self.pop_size - 1:
                    parent_1 = self.parents[index]
                    children.append(copy.deepcopy(parent_1))
                    index += 1
                # for two parents, product two replacement children
                else:
                    # select two parents
                    parent_1 = self.parents[index]
                    parent_2 = self.parents[index + 1]
                    # print(parent_1)
                    # print(parent_2)
                    child_1 = copy.deepcopy(parent_1)
                    child_2 = copy.deepcopy(parent_2)
                    # perform uniform crossover on all genes
                    # for row_idx in range(len(parent_1)):
                    #     for col_idx in range(len(parent_1[row_idx])):
                    #         # decide if swap values for the gene
                    #         if self.coin_toss > random.random():
                    #             child_1[row_idx][col_idx] == child_2[row_idx][col_idx]
                    #             child_2[row_idx][col_idx] == child_1[row_idx][col_idx]
                    for (item1, item2) in zip(child_1, child_2):
                        for row_idx in range(len(item1)):
                            for col_idx in range(len(item1[row_idx])):
                                # decide if swap values for the gene
                                if self.coin_toss > random.random():
                                    item1[row_idx][col_idx] == item2[row_idx][col_idx]
                                    item2[row_idx][col_idx] == item1[row_idx][col_idx]
                    children.extend([child_1, child_2])
                    index += 2
                    # if self.verbose == True:
                        # print("Parent:",parent_1)
                        # print("Child1:",child_1)
                    # sys.exit(0)
            # according to crossover probability use parent instead of crossover
            else:
                # if only one parent left, use parent as one replacement child
                if index >= self.pop_size - 1:
                    parent_1 = self.parents[index]
                    children.append(copy.deepcopy(parent_1))
                    index += 1
                else:
                    # select two parents
                    parent_1 = self.parents[index]
                    parent_2 = self.parents[index + 1]
                    # if self.verbose == True:
                        # print(f"Don't Perform Crossover")
                    children.extend([copy.deepcopy(parent_1), copy.deepcopy(parent_2)])
                    index += 2

        return children

    # use a mutation probability to do mutation and add a tunable weight to each weight in our children
    def mutation(self, children):
        # for each gene on each child, use mutation probabilty to check if gene mutates
        for child in children:
            for item in child:
                for row_idx in range(len(item)):
                    for col_idx in range(len(item[row_idx])):
                        if self.pm >= random.random():
                            # mutate selected gene
                            item[row_idx][col_idx] += random.uniform(0,self.mutate_sigma)
        
        return children

    # generational replacement, replace all old generation with all children
    def replacement(self, children):
        self.population = copy.deepcopy(children)  
        
    # n replacement, replace all n old generation with all n children
    def n_replacement(self, children):
        replace = []
        # if self.verbose == True:
            # print("Tournament pool:")
        while len(replace) != self.n_size:
            # if self.verbose == True:
                # print(f"round {len(replace)}")
            # randomly select k participants from the population
            # tournament_pool = random.choices(self.population, k=self.tournament_size)
            # fit_keys = list(self.fitness_dict.keys())
            tournament_pool = random.sample(list(range(0,self.pop_size)), k=self.tournament_size)
            # tournament_pool = random.choices(list(range(0,self.pop_size)), k=self.tournament_size)
            pool_worst = tournament_pool[0]
            # if self.verbose == True:
                # print("Tournament pool:")
                # print(tournament_pool,",", end="")
            # select pool participant with the best fitness as parent
            for p in tournament_pool:
                # if self.verbose == True:
                    # print("worst",pool_worst,":",self.fitness_dict[pool_worst]," , ","current",p,":",self.fitness_dict[p])
                # if p fitness > pool_best fitness:
                if self.version == "class":
                    if self.fitness_dict[p] < self.fitness_dict[pool_worst]:
                        # if self.verbose == True:
                            # print("New Best",p,"!")
                        pool_worst = p
                elif self.version == "regress":
                    if self.fitness_dict[p] > self.fitness_dict[pool_worst]:
                        # if self.verbose == True:
                            # print("New Best",p,"!")
                        pool_worst = p
            replace.append(pool_worst)
            # if self.verbose == True:
                # print("worst:",pool_worst,"\n")
        i = 0
        # if self.verbose == True:
            # print(replace)
        for r in replace:
            self.population[r] = copy.deepcopy(children[i])
            i += 1