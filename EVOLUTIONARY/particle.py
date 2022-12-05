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
    def __init__(self, x0, shape1,  shape2, size, classes, mlp, testoutput) -> None:
        # particle position
        self.position_i = []
        # particle velocity
        self.velocity_i = []
        # best personal position 
        self.pos_best_i = []
        self.f1_best_i = 0
        self.f1_i = 0
        self.classes = classes
        self.mlp = mlp
        self.testoutput = testoutput
        self.size = size  
        for i in range(0, self.size):
            self.position_i.append(x0[i])
            vel = []
            vel.append(np.full(shape1, random.uniform(-1, 1)))
            vel.append(np.full(shape2, random.uniform(-1, 1)))
            self.velocity_i.append(vel)
    
    def fitness(self):
        # check to see the current best is an indivisual best or not
        for i in range(0, len(self.position_i)):
            result = UTILS.get_performance(UTILS, self.mlp, self.position_i[i], self.classes, self.testoutput)
            loss = UTILS.calculate_loss_np(UTILS, result, self.classes)
            self.f1_i = loss["F1"]
            if (self.f1_i > self.f1_best_i):
                self.pos_best_i = self.position_i
                self.f1_best_i = self.f1_i
    
    def update_velocity(self, pos_best_g):
        # constant inertia weight
        w = 1
        c1 = 1 # congative constat
        c2 = 2 # social constant
        for i in range(0, self.size):
            r1 = random.random()
            r2 = random.random()
            conative_vel = c1*r1*(np.subtract(self.pos_best_i[i], self.position_i[i]))
            social_vel = c2*r2*(np.subtract(pos_best_g[i], self.position_i[i]))
            self.velocity_i[i] = w*self.velocity_i[i] + conative_vel + social_vel

    def update_position(self):
        for i in range(0, self.size):
            self.position_i[i] = self.position_i[i] + self.velocity_i[i]




        

 