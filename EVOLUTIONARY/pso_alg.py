##########################################################################
# Xuying Swift and Christine Johnson, 11/2022, CSCI 447 Machine Learning
# Implements a gbest Particle Swarm Optimization Algorithm
##########################################################################

import math
import numpy as np
import random

from utils import UTILS

def objective_fuction(x, y, i, j):
    classes = [1, 2, 3, 4, 5, 6, 7]
    loss = {}
    # given a weight matrix, we need to calculate the best performance 
    loss_x = UTILS.calculate_loss_np(UTILS, x, classes)
    loss_y = UTILS.calculate_loss_np(UTILS, y, classes)
    x_list = []
    y_list = []
    x_list.append(i)
    x_list.append(loss_x["F1"])
    y_list.append(j)
    y_list.append(loss_y["F1"])
    loss["x"] = x_list
    loss["y"] = y_list
    return loss 
   

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
        self.pbest = self.X
        self.i = 0
        self.j = 1
        self.pbest_obj = objective_fuction(self.X[self.i], self.X[self.j], self.i, self.j)
        self.gbest = None
        self.gbest_obj = self.find_max_loss(self.pbest_obj)
       
        # self.gbest_obj = self.pbest_obj.min()

    def find_max_index(self, loss):
        max_index = 0
        max_f1 = 0
        for value in loss.values():
            if value[1] >  max_f1:
                max_f1= value[1]
        for value in loss.values():
            if max_f1 == value[1]:
                max_index = value[0]
        return max_index

    def find_max_F1(self, loss):
        max_f1 = 0
        for value in loss.values():
            if value[1] >  max_f1:
                max_f1= value[1]
        return max_f1

    def find_max_loss(self, loss):
        max_f1 = 0
        target = {}
        for value in loss.values():
            if value[1] > max_f1:
                max_f1 = value[1]
        
        for key, value in loss.items():
            if value[1] == max_f1:
                target[key] = value
        return target

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
            r1, r2 = np.random.rand(2)
            index = self.find_max_index(self.pbest_obj)
            if (index == 0):
                self.gbest = self.pbest[index]
            else:
                self.gbest = self.pbest[:index]
            #print(self.gbest_obj)
            
            #print(r1, r2)
            #print(self.V)

            pbest_subtracted = list()
            # print("print out p_best")
            # print(self.pbest)
            # print("print out X")
            # print(self.X)
            for item1, item2 in zip(self.pbest, self.X):
                item = item1 - item2
                pbest_subtracted.append(item)
            #print(pbest_subtracted)

            gbest_subtracted = list()
            for item1, item2 in zip(self.gbest, self.X):
                item = item1 - item2
                gbest_subtracted.append(item)
            #print(gbest_subtracted)

            frist_part = [i * self.w for i in self.V]
            second_part = [i * self.c1*r1 for i in pbest_subtracted]
            third_part = [i * self.c2*r2 for i in gbest_subtracted]
            self.V =  frist_part + second_part + third_part
            #print(self.V)
            #print("NEW X")
            self.X = self.X + self.V
            #print(self.X)
            obj = objective_fuction(self.X[self.i], self.X[self.j], self.i, self.j)
            #print(obj)
            max_index = self.find_max(self.pbest_obj, obj)
            self.pbest[:max_index] = self.X[:max_index]
            pbest_index = self.find_max_index(self.pbest_obj)
            self.gbest = self.pbest[:pbest_index]
            self.gbest_obj = self.find_max_loss(self.pbest_obj)
            print(self.gbest_obj)
        