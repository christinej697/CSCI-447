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
# print(total_classes)

class_two = 0
class_four = 0

class_two = last_n_column.value_counts()[2]
class_four = last_n_column.value_counts()[4]

# print(class_two)
# print(class_four)

p_class_two = class_two / total_classes
p_class_four = class_four / total_classes

# print(p_class_two)
# print(p_class_four)

dataset = dataset.replace("?", np.NaN)
dataset = dataset.dropna()


dataset.columns = ["Sample code number", "Uniformity of Clump Thickness", "Uniformity of Cell Size", "Cell Shape", "Marginal Adhesion", "Single Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses", "Class"]
dataset = dataset.sort_values(by=['Class'])

clump_df = dataset.groupby(["Class", "Uniformity of Clump Thickness"]).size().reset_index(name = "total")
# print(clump_df)
one = clump_df.iat[0,2]

dd = dataset.groupby(["Class"]).size()
ff = dd.reset_index(name = "total")
# print(ff)
print(dataset)

feature = (one +1) / class_two + (len(dataset.columns) - 1)
# print(feature)
#file_name_1 = "glass.data"
#df = pd.read_csv(file_name_1, sep=",")
#df = df.replace("?", np.NaN)
#df = df.dropna()
#df.columns = ["Id Number", "RI: Retractive Index", "Na: Sodium", "Mg: Magnesium", "Al: Aluminum", "Si: Silicon", "K: Potassium", "Ca: Calcium", "Ba: Barium", "Fe: Iron", "Class"]
#print(df.shape)
f1_df = dataset.groupby(["Class", "Uniformity of Clump Thickness"]).size().reset_index(name = "total")
f1_2 = f1_df[f1_df['Class'] == 2]
f1_4 = f1_df[f1_df['Class'] == 4]
print(f1_df)
# print(f1_2)
# print(f1_4)
f2_df = dataset.groupby(["Class", "Uniformity of Cell Size"]).size().reset_index(name = "total")
# f2_2 = f1_df[f1_df['Class'] == 2]
# f2_4 = f1_df[f1_df['Class'] == 4]
print(f2_df)
f3_df = dataset.groupby(["Class", "Cell Shape"]).size().reset_index(name = "total")
# f3_2 = f1_df[f1_df['Class'] == 2]
# f3_4 = f1_df[f1_df['Class'] == 4]
print(f3_df)
f4_df = dataset.groupby(["Class", "Marginal Adhesion"]).size().reset_index(name = "total")
# f4_2 = f1_df[f1_df['Class'] == 2]
# f4_4 = f1_df[f1_df['Class'] == 4]
print(f4_df)
f5_df = dataset.groupby(["Class", "Single Epithelial Cell Size"]).size().reset_index(name = "total")
# f5_2 = f1_df[f1_df['Class'] == 2]
# f5_4 = f1_df[f1_df['Class'] == 4]
print(f5_df)
f6_df = dataset.groupby(["Class", "Bare Nuclei"]).size().reset_index(name = "total")
# f6_2 = f1_df[f1_df['Class'] == 2]
# f6_4 = f1_df[f1_df['Class'] == 4]
print(f6_df)
f7_df = dataset.groupby(["Class", "Bland Chromatin"]).size().reset_index(name = "total")
# f7_2 = f1_df[f1_df['Class'] == 2]
# f7_4 = f1_df[f1_df['Class'] == 4]
print(f7_df)
f8_df = dataset.groupby(["Class", "Normal Nucleoli"]).size().reset_index(name = "total")
# f8_2 = f1_df[f1_df['Class'] == 2]
# f8_4 = f1_df[f1_df['Class'] == 4]
print(f8_df)
f9_df = dataset.groupby(["Class", "Mitoses"]).size().reset_index(name = "total")
# f9_2 = f1_df[f1_df['Class'] == 2]
# f9_4 = f1_df[f1_df['Class'] == 4]
print(f9_df)
# print(f9_df.shape)

# check2 = f1_df.groupby(['class']).size()
# for row in f1_2
f1_2 = dataset[(dataset['Class'] == 2) & (dataset['Uniformity of Clump Thickness'] == 1)]
f2_2 = dataset[(dataset['Class'] == 2) & (dataset['Uniformity of Clump Thickness'] == 2)]
print("Sum: ",f1_2.shape[0])
# f1_2 = dataset[(dataset['Class'] == 2)]
# print("here")
print("Sum: ",f2_2.shape)
print("There")

# Create Cancer Likelihood Table
data = [
        [dataset[(dataset['Class'] == 2) & (dataset['Uniformity of Clump Thickness'] == 1)].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Uniformity of Cell Size'] == 1)].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Cell Shape'] == 1)].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Marginal Adhesion'] == 1)].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Single Epithelial Cell Size'] == 1)].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Bare Nuclei'] == '1')].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Bland Chromatin'] == 1)].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Normal Nucleoli'] == 1)].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Mitoses'] == 1)].shape[0]],
        [dataset[(dataset['Class'] == 2) & (dataset['Uniformity of Clump Thickness'] == 2)].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Uniformity of Cell Size'] == 2)].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Cell Shape'] == 2)].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Marginal Adhesion'] == 2)].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Single Epithelial Cell Size'] == 2)].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Bare Nuclei'] == '2')].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Bland Chromatin'] == 2)].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Normal Nucleoli'] == 2)].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Mitoses'] == 2)].shape[0]],
        [dataset[(dataset['Class'] == 2) & (dataset['Uniformity of Clump Thickness'] == 3)].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Uniformity of Cell Size'] == 3)].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Cell Shape'] == 3)].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Marginal Adhesion'] == 3)].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Single Epithelial Cell Size'] == 3)].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Bare Nuclei'] == '3')].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Bland Chromatin'] == 3)].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Normal Nucleoli'] == 3)].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Mitoses'] == 3)].shape[0]],
        [dataset[(dataset['Class'] == 2) & (dataset['Uniformity of Clump Thickness'] == 4)].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Uniformity of Cell Size'] == 4)].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Cell Shape'] == 4)].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Marginal Adhesion'] == 4)].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Single Epithelial Cell Size'] == 4)].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Bare Nuclei'] == '4')].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Bland Chromatin'] == 4)].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Normal Nucleoli'] == 4)].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Mitoses'] == 4)].shape[0]],
        [dataset[(dataset['Class'] == 2) & (dataset['Uniformity of Clump Thickness'] == 5)].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Uniformity of Cell Size'] == 5)].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Cell Shape'] == 5)].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Marginal Adhesion'] == 5)].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Single Epithelial Cell Size'] == 5)].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Bare Nuclei'] == '5')].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Bland Chromatin'] == 5)].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Normal Nucleoli'] == 5)].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Mitoses'] == 5)].shape[0]],
        [dataset[(dataset['Class'] == 2) & (dataset['Uniformity of Clump Thickness'] == 6)].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Uniformity of Cell Size'] == 6)].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Cell Shape'] == 6)].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Marginal Adhesion'] == 6)].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Single Epithelial Cell Size'] == 6)].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Bare Nuclei'] == '6')].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Bland Chromatin'] == 6)].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Normal Nucleoli'] == 6)].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Mitoses'] == 6)].shape[0]],
        [dataset[(dataset['Class'] == 2) & (dataset['Uniformity of Clump Thickness'] == 7)].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Uniformity of Cell Size'] == 7)].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Cell Shape'] == 7)].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Marginal Adhesion'] == 7)].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Single Epithelial Cell Size'] == 7)].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Bare Nuclei'] == '7')].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Bland Chromatin'] == 7)].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Normal Nucleoli'] == 7)].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Mitoses'] == 7)].shape[0]],
        [dataset[(dataset['Class'] == 2) & (dataset['Uniformity of Clump Thickness'] == 8)].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Uniformity of Cell Size'] == 8)].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Cell Shape'] == 8)].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Marginal Adhesion'] == 8)].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Single Epithelial Cell Size'] == 8)].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Bare Nuclei'] == '8')].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Bland Chromatin'] == 8)].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Normal Nucleoli'] == 8)].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Mitoses'] == 8)].shape[0]],
        [dataset[(dataset['Class'] == 2) & (dataset['Uniformity of Clump Thickness'] == 9)].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Uniformity of Cell Size'] == 9)].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Cell Shape'] == 9)].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Marginal Adhesion'] == 9)].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Single Epithelial Cell Size'] == 9)].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Bare Nuclei'] == '9')].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Bland Chromatin'] == 9)].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Normal Nucleoli'] == 9)].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Mitoses'] == 9)].shape[0]],
        [dataset[(dataset['Class'] == 2) & (dataset['Uniformity of Clump Thickness'] == 10)].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Uniformity of Cell Size'] == 10)].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Cell Shape'] == 10)].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Marginal Adhesion'] == 10)].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Single Epithelial Cell Size'] == 10)].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Bare Nuclei'] == '10')].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Bland Chromatin'] == 10)].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Normal Nucleoli'] == 10)].shape[0],dataset[(dataset['Class'] == 2) & (dataset['Mitoses'] == 10)].shape[0]],
        [dataset[(dataset['Class'] == 4) & (dataset['Uniformity of Clump Thickness'] == 1)].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Uniformity of Cell Size'] == 1)].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Cell Shape'] == 1)].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Marginal Adhesion'] == 1)].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Single Epithelial Cell Size'] == 1)].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Bare Nuclei'] == '1')].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Bland Chromatin'] == 1)].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Normal Nucleoli'] == 1)].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Mitoses'] == 1)].shape[0]],
        [dataset[(dataset['Class'] == 4) & (dataset['Uniformity of Clump Thickness'] == 2)].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Uniformity of Cell Size'] == 2)].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Cell Shape'] == 2)].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Marginal Adhesion'] == 2)].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Single Epithelial Cell Size'] == 2)].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Bare Nuclei'] == '2')].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Bland Chromatin'] == 2)].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Normal Nucleoli'] == 2)].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Mitoses'] == 2)].shape[0]],
        [dataset[(dataset['Class'] == 4) & (dataset['Uniformity of Clump Thickness'] == 3)].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Uniformity of Cell Size'] == 3)].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Cell Shape'] == 3)].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Marginal Adhesion'] == 3)].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Single Epithelial Cell Size'] == 3)].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Bare Nuclei'] == '3')].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Bland Chromatin'] == 3)].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Normal Nucleoli'] == 3)].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Mitoses'] == 3)].shape[0]],
        [dataset[(dataset['Class'] == 4) & (dataset['Uniformity of Clump Thickness'] == 4)].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Uniformity of Cell Size'] == 4)].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Cell Shape'] == 4)].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Marginal Adhesion'] == 4)].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Single Epithelial Cell Size'] == 4)].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Bare Nuclei'] == '4')].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Bland Chromatin'] == 4)].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Normal Nucleoli'] == 4)].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Mitoses'] == 4)].shape[0]],
        [dataset[(dataset['Class'] == 4) & (dataset['Uniformity of Clump Thickness'] == 5)].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Uniformity of Cell Size'] == 5)].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Cell Shape'] == 5)].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Marginal Adhesion'] == 5)].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Single Epithelial Cell Size'] == 5)].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Bare Nuclei'] == '5')].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Bland Chromatin'] == 5)].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Normal Nucleoli'] == 5)].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Mitoses'] == 5)].shape[0]],
        [dataset[(dataset['Class'] == 4) & (dataset['Uniformity of Clump Thickness'] == 6)].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Uniformity of Cell Size'] == 6)].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Cell Shape'] == 6)].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Marginal Adhesion'] == 6)].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Single Epithelial Cell Size'] == 6)].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Bare Nuclei'] == '6')].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Bland Chromatin'] == 6)].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Normal Nucleoli'] == 6)].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Mitoses'] == 6)].shape[0]],
        [dataset[(dataset['Class'] == 4) & (dataset['Uniformity of Clump Thickness'] == 7)].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Uniformity of Cell Size'] == 7)].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Cell Shape'] == 7)].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Marginal Adhesion'] == 7)].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Single Epithelial Cell Size'] == 7)].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Bare Nuclei'] == '7')].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Bland Chromatin'] == 7)].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Normal Nucleoli'] == 7)].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Mitoses'] == 7)].shape[0]],
        [dataset[(dataset['Class'] == 4) & (dataset['Uniformity of Clump Thickness'] == 8)].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Uniformity of Cell Size'] == 8)].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Cell Shape'] == 8)].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Marginal Adhesion'] == 8)].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Single Epithelial Cell Size'] == 8)].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Bare Nuclei'] == '8')].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Bland Chromatin'] == 8)].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Normal Nucleoli'] == 8)].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Mitoses'] == 8)].shape[0]],
        [dataset[(dataset['Class'] == 4) & (dataset['Uniformity of Clump Thickness'] == 9)].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Uniformity of Cell Size'] == 9)].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Cell Shape'] == 9)].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Marginal Adhesion'] == 9)].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Single Epithelial Cell Size'] == 9)].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Bare Nuclei'] == '9')].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Bland Chromatin'] == 9)].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Normal Nucleoli'] == 9)].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Mitoses'] == 9)].shape[0]],
        [dataset[(dataset['Class'] == 4) & (dataset['Uniformity of Clump Thickness'] == 10)].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Uniformity of Cell Size'] == 10)].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Cell Shape'] == 10)].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Marginal Adhesion'] == 10)].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Single Epithelial Cell Size'] == 10)].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Bare Nuclei'] == '10')].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Bland Chromatin'] == 10)].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Normal Nucleoli'] == 10)].shape[0],dataset[(dataset['Class'] == 4) & (dataset['Mitoses'] == 10)].shape[0]]]

cancer_probs = pd.DataFrame(data, columns=["Uniformity of Clump Thickness", "Uniformity of Cell Size", "Cell Shape", "Marginal Adhesion", "Single Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses"], index=['1-2','2-2','3-2','4-2','5-2','6-2','7-2','8-2','9-2','10-2','1-4','2-4','3-4','4-4','5-4','6-4','7-4','8-4','9-4','10-4'])

print(cancer_probs)
print('Bare Nuclei')
print(f6_df)
# print(dataset['Bare Nuclei'])
# print('Bare Nuclei')
# print(dataset[(dataset['Class'] == 4) & (dataset['Bare Nuclei'] == '10')])
# print('Bare Nuclei')
# print(dataset[dataset['Bare Nuclei'] == 10])
# print('Bare Nuclei')
# print(dataset[dataset['Bare Nuclei'] == '10'])

def f_prob(a_c, n_c, d):
    return (a_c + 1)/(n_c + d)
