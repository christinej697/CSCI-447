from ast import Delete
from ctypes import sizeof
from os import remove
import pandas as pd
import numpy as np
import scipy as sp
file_name = "breast-cancer-wisconsin.data"
dataset = pd.read_csv(file_name, sep=",")
dataset.dropna()
cols = dataset.columns.to_list()
N = len(cols)
last_n_column  = dataset.iloc[: , -1:]

total_classes = last_n_column.values.size
print(total_classes)

class_two = 0
class_four = 0

class_two = last_n_column.value_counts()[2]
class_four = last_n_column.value_counts()[4]

print(class_two)
print(class_four)

p_class_two = class_two / total_classes
p_class_four = class_four / total_classes

print(p_class_two)
print(p_class_four)

dataset = dataset.replace("?", np.NaN)
dataset = dataset.dropna()


dataset.columns = ["Sample code number", "Uniformity of Clump Thickness", "Uniformity of Cell Size", "Cell Shape", "Marginal Adhesion", "Single Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses", "Class"]
dataset = dataset.sort_values(by=['Class'])

clump_df = dataset.groupby(["Class", "Uniformity of Clump Thickness"]).size().reset_index(name = "total")
print(clump_df)
one = clump_df.iat[0,2]

dd = dataset.groupby(["Class"]).size()
ff = dd.reset_index(name = "total")
print(ff)

feature = (one +1) / class_two + (len(dataset.columns) - 1)
print(feature)
#file_name_1 = "glass.data"
#df = pd.read_csv(file_name_1, sep=",")
#df = df.replace("?", np.NaN)
#df = df.dropna()
#df.columns = ["Id Number", "RI: Retractive Index", "Na: Sodium", "Mg: Magnesium", "Al: Aluminum", "Si: Silicon", "K: Potassium", "Ca: Calcium", "Ba: Barium", "Fe: Iron", "Class"]
#print(df.shape)
        





   

