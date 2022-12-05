import math
import numpy as np
import random
from utils import UTILS

def obj_function(x):
    # get the best F1 score
    max_f1 = 0
    classes = [1, 2, 3, 4, 5, 6, 7]
    # given a weight matrix, we need to calculate the best performance 
    loss = UTILS.calculate_loss_np(UTILS, x, classes)
    if loss["F1"] > max_f1:
        max_f1 = loss["F1"]
    return max_f1

class Particle:
    def __init__(self, x0, shape, num_dimensions, classes, mlp, testoutput) -> None:
        # particle position
        self.position_i = []
        # particle velocity
        self.velocity_i = np.full(shape, random.uniform(-1, 1))
        # best personal position 
        self.pos_best_i = []
        self.err_best_i = -1
        self.err_i = -1
        self.num_dimensions = shape
        self.classes = classes
        self.mlp = mlp
        self.testoutput = testoutput    
        for i in range(0, num_dimensions):
            self.position_i.append(x0[i])
    
    def fitness(self):
        # check to see the current best is an indivisual best or not
        for i in range(0, len(self.position_i)):
            result = UTILS.get_performance(UTILS, self.mlp, self.position_i[i], self.classes, self.testoutput)
            print(result)
            self.err_i = UTILS.calculate_loss_np(UTILS, self.position_i[i], self.classes)
            if (self.err_i > self.err_best_i):
                self.err_best_i = self.pos_best_i
                self.err_best_i = self.err_i
    
    def update_velocity(self, pos_best_g):
        # constant inertia weight
        w = 0.5
        c1 = 1 # congative constat
        c2 = 2 # social constant

        for i in range(0, self.num_dimensions):
            r1 = random.random()
            r2 = random.random()

            conative_vel = c1*r1*(self.pos_best_i[i] - self.position_i[i])
            social_vel = c2*r2*(pos_best_g[i] - self.position_i[i])
            self.velocity_i[i] = w* self.velocity_i[i] + conative_vel + social_vel

    def update_position(self, bounds):
        for i in range(0, self.num_dimensions):
            self.position_i[i] = self.position_i[i] + self.velocity_i[i]

            if self.position_i[i] > bounds[i][0]:
                self.position_i[i] = bounds[i][0]
            
            if self.position_i[i] < bounds[i][0]:
                self.position_i[i] = bounds[i][0]




        

 