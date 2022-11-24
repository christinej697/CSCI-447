##########################################################################
# Xuying Swift and Christine Johnson, 11/2022, CSCI 447 Machine Learning
# Implements a gbest Particle Swarm Optimization Algorithm
##########################################################################

import numpy as np
import random
from utils import UTILS

class PSO:
    def __init__(self, version: str, population, classes = None, verbose: str = None):
        self.version = version
        self.population = population
        self.pop_size = len(population)
        self.classes = classes
        self.verbose = verbose
        self.fitness_dict = {}
        self.fit_keys = []

    def run(self):
        pass