##########################################################################
# Xuying Swift and Christine Johnson, 11/2022, CSCI 447 Machine Learning
# Implements a gbest Particle Swarm Optimization Algorithm
##########################################################################

import math
import numpy as np
import random
from utils import UTILS   

class PSO:
    def __init__(self, population, c1: float, c2: float, w: float, n_particals: int):
   
        self.population = population
        self.pop_size = len(population)
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.n_particals = n_particals
        self.X = random.sample(self.population, self.n_particals)
        self.V = [i * 0 for i in self.X]


    def find_max(self, dict1, dict2):
        max_index = 0
        dict1_f1 = self.find_max_F1(dict1)
        dict1_index = self.find_max_index(dict1)
        dict2_f1 = self.find_max_F1(dict2)
        dict2_index = self.find_max_F1(dict2)
        if (dict1_f1 > dict2_f1):
            max_index = dict1_index
        else:
            max_index = dict2_index
        return max_index
        
    #Function to do one iteration of particle swarm optimization
    def run(self):
        for i in range(self.n_particals):
            pass