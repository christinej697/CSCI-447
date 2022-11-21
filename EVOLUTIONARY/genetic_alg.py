##########################################################################
# Xuying Swift and Christine Johnson, 11/2022, CSCI 447 Machine Learning
# Implements a Genetic Algorithm with a steady state population, tournament selection,
# uniform crossover, and generational replacement
##########################################################################

import numpy as np
import random
class GA:
    def __init__(self, version: str, population, num_generations: int = 3, tournament_size: int = 3, crossover_probability: float = 0.9, coin_toss: float = 0.5, mutation_probability: float = 0.05, mutate_sigma: float = 0.1, verbose: str = None):
        self.version = version
        self.population = population
        self.pop_size = len(population)
        self.num_generations = num_generations
        self.tournament_size = tournament_size
        self.pr = crossover_probability
        self.coin_toss = coin_toss
        self.pm = mutation_probability
        self.mutate_sigma = mutate_sigma
        self.verbose = verbose


    # initialize the population. And the population is a single fold test output from MLP network with the best performance

    # evaluate all populations fitnesss, we rank their fitness by best to worst

    # use tournament method: select k random participants from the population, then the best of the k
    # use above to produce and return all parents for generational replacement
    def selection(self):
        parents = []
        while len(parents) != self.pop_size:
            # randomly select k participants from the population
            tournament_pool = random.choices(self.population, k=self.tournament_size)
            pool_best = tournament_pool[0]
            # select pool participant with the best fitness as parent
            for p in tournament_pool:
                # if p fitness > pool_best fitness:
                    #current_best = p
                pass
            parents.append[pool_best]
        return parents

    # use uniform crossover to produce next generation from parents, w/ crossover probability
    def crossover(self, parents):
        children = []
        index = 0
        while len(children) != self.pop_size:
            # select two parents
            parent_1 = parents[index]
            parent_2 = parents[index + 1]
            child_1 = parent_1
            child_2 = parent_2
            # check if we will perform crossover acccording to crossover probability
            if random.random() < self.pr:
                # if only one parent left, use parent as one replacement child
                if index == self.pop_size - 1:
                    children.append
                # for two parents, product two replacement children
                else:
                    # perform uniform crossover on all genes
                    for row_idx in range(len(parent_1)):
                        for col_idx in range(len(parent_1[row_idx])):
                            # decide if swap values for the gene
                            if self.coin_toss > random.random():
                                child_1[row_idx][col_idx] == child_2[row_idx][col_idx]
                                child_2[row_idx][col_idx] == child_1[row_idx][col_idx]
                    children.extend(child_1, child_2)
            # according to crossover probability use parent instead of crossover
            else:
                children.extend(child_1, child_2)

        return children

    # use a mutation probability to do mutation and add a tunable weight to each weight in our children
    def mutation(self, children):
        # for each child, use mutation probabilty to check if child mutates
        for child in children:
            if self.pm > random.random():
                row_idx = random.randrange(len(child))
                col_idx = random.randrange(len(child[row_idx]))
                # mutate selected gene
                child[row_idx][col_idx] += random.uniform(0,self.mutate_sigma)
        
        return children

    # generational replacement, replace all n old generation with all n children
    def replacement(self, children):
        self.population = children

    # termination: a set number of generations or until performance is not improving anymore