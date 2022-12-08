import math
import numpy as np
import random
from utils import UTILS

def obj_function(x):
    # get the best F1 score
    max_performance = 0
    # given a weight matrix, we need to calculate the best performance 
    classes = [1, 2, 3, 4, 5, 6, 7]
    loss = UTILS.calculate_loss_np(UTILS, x, classes)
    if loss["F1"] > max_performance:
        max_performance = loss["F1"]
    return max_performance

class Particle:
    def __init__(self, x0, size, classes, mlp, testoutput, version, shape1,  shape2 = None, shape3 = None, xmax = None, xmin = None) -> None:
        # particle position
        self.position_i = []
        # particle velocity
        self.velocity_i = []
        # best personal position 
        self.pos_best_i = []
        self.perform_best_i = 0
        self.perform_i = 0
        self.classes = classes
        self.mlp = mlp
        self.testoutput = testoutput
        self.size = size  
        self.version = version
        self.x_max = xmax
        self.x_min = xmin
        self.loss_i = None
        self.loss_best = None
        for i in range(0, self.size):
            self.position_i.append(x0[i])
            vel = []
            vel.append(np.full(shape1, random.uniform(-1, 1)))
            if shape2 != None:
                vel.append(np.full(shape2, random.uniform(-1, 1)))
            if shape3 != None:
                vel.append(np.full(shape3, random.uniform(-1, 1)))
            self.velocity_i.append(vel)
    # check to see the current best is an indivisual best or not
    def fitness(self):
        for i in range(0, len(self.position_i)):
            result = UTILS.get_performance(UTILS, self.mlp, self.position_i[i], self.classes, self.testoutput)
            if self.version == "class":
                loss = UTILS.calculate_loss_np(UTILS, result, self.classes)
                self.perform_i = loss["Accuracy"]
                self.loss_i = loss
            elif self.version == "regress":
                loss = UTILS.calculate_loss_for_regression(UTILS, result, self.test_values, self.x_max, self.x_min)
                self.perform_i = loss["MSE"]
                self.loss_i = loss
            if (self.perform_i > self.perform_best_i):
                self.pos_best_i = self.position_i
                self.perform_best_i = self.perform_i

    # function created to update the velocity for particales
    def update_velocity(self, pos_best_g):
        # constant inertia weight
        w = 1
        # congative constant
        c1 = 1
        # social constant
        c2 = 2 
        for i in range(0, self.size):
            print("~~~~~~~~~~~~~updating the velocity~~~~~~~~~~~~~~")
            r1 = random.random()
            r2 = random.random()
            cognative_vel = c1*r1*(np.subtract(self.pos_best_i[i], self.position_i[i]))
            social_vel = c2*r2*(np.subtract(pos_best_g[i], self.position_i[i]))
            print("Previous velocity: ", self.velocity_i[i])
            self.velocity_i[i] = w*self.velocity_i[i] + cognative_vel + social_vel
            print("Upaded velocity: ", self.velocity_i[i])

      
           

    # function created to update the position for particles.
    def update_position(self):
        for i in range(0, self.size):
            self.position_i[i] = self.position_i[i] + self.velocity_i[i]
            print("~~~~~~~~~~~~~~update positons~~~~~~~~~~~~~~~~~~~~~")
            print("At position i in the population: ", i)
            print("With the new speed means added velocity at each position: ", self.position_i[i])




        

 