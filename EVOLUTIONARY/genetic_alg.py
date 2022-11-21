import numpy as np
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

    def crossover(Self):
        pass