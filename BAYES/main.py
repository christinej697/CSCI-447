from ast import Delete
from ctypes import sizeof
from os import remove
import pandas as pd
import numpy as np
import scipy as sp
file_name = "breast-cancer-wisconsin.data"
cancer_dataset = pd.read_csv(file_name, sep=",")
cancer_dataset.dropna()
cols = cancer_dataset.columns.to_list()
N = len(cols)
last_n_column  = cancer_dataset.iloc[: , -1:]

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

cancer_dataset = cancer_dataset.replace("?", np.NaN)
cancer_dataset = cancer_dataset.dropna()


cancer_dataset.columns = ["Sample code number", "Uniformity of Clump Thickness", "Uniformity of Cell Size", "Cell Shape", "Marginal Adhesion", "Single Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses", "Class"]
cancer_dataset = cancer_dataset.sort_values(by=['Class'])

clump_df = cancer_dataset.groupby(["Class", "Uniformity of Clump Thickness"]).size().reset_index(name = "total")
# print(clump_df)
one = clump_df.iat[0,2]

dd = cancer_dataset.groupby(["Class"]).size()
ff = dd.reset_index(name = "total")
# print(ff)
print(cancer_dataset)

feature = (one +1) / class_two + (len(cancer_dataset.columns) - 1)
# print(feature)
#file_name_1 = "glass.data"
#df = pd.read_csv(file_name_1, sep=",")
#df = df.replace("?", np.NaN)
#df = df.dropna()
#df.columns = ["Id Number", "RI: Retractive Index", "Na: Sodium", "Mg: Magnesium", "Al: Aluminum", "Si: Silicon", "K: Potassium", "Ca: Calcium", "Ba: Barium", "Fe: Iron", "Class"]
#print(df.shape)
f1_df = cancer_dataset.groupby(["Class", "Uniformity of Clump Thickness"]).size().reset_index(name = "total")
f1_2 = f1_df[f1_df['Class'] == 2]
f1_4 = f1_df[f1_df['Class'] == 4]
print(f1_df)
# print(f1_2)
# print(f1_4)
f2_df = cancer_dataset.groupby(["Class", "Uniformity of Cell Size"]).size().reset_index(name = "total")
# f2_2 = f1_df[f1_df['Class'] == 2]
# f2_4 = f1_df[f1_df['Class'] == 4]
print(f2_df)
f3_df = cancer_dataset.groupby(["Class", "Cell Shape"]).size().reset_index(name = "total")
# f3_2 = f1_df[f1_df['Class'] == 2]
# f3_4 = f1_df[f1_df['Class'] == 4]
print(f3_df)
f4_df = cancer_dataset.groupby(["Class", "Marginal Adhesion"]).size().reset_index(name = "total")
# f4_2 = f1_df[f1_df['Class'] == 2]
# f4_4 = f1_df[f1_df['Class'] == 4]
print(f4_df)
f5_df = cancer_dataset.groupby(["Class", "Single Epithelial Cell Size"]).size().reset_index(name = "total")
# f5_2 = f1_df[f1_df['Class'] == 2]
# f5_4 = f1_df[f1_df['Class'] == 4]
print(f5_df)
f6_df = cancer_dataset.groupby(["Class", "Bare Nuclei"]).size().reset_index(name = "total")
# f6_2 = f1_df[f1_df['Class'] == 2]
# f6_4 = f1_df[f1_df['Class'] == 4]
print(f6_df)
f7_df = cancer_dataset.groupby(["Class", "Bland Chromatin"]).size().reset_index(name = "total")
# f7_2 = f1_df[f1_df['Class'] == 2]
# f7_4 = f1_df[f1_df['Class'] == 4]
print(f7_df)
f8_df = cancer_dataset.groupby(["Class", "Normal Nucleoli"]).size().reset_index(name = "total")
# f8_2 = f1_df[f1_df['Class'] == 2]
# f8_4 = f1_df[f1_df['Class'] == 4]
print(f8_df)
f9_df = cancer_dataset.groupby(["Class", "Mitoses"]).size().reset_index(name = "total")
# f9_2 = f1_df[f1_df['Class'] == 2]
# f9_4 = f1_df[f1_df['Class'] == 4]
print(f9_df)
# print(f9_df.shape)

# check2 = f1_df.groupby(['class']).size()
# for row in f1_2
f1_2 = cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Uniformity of Clump Thickness'] == 1)]
f2_2 = cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Uniformity of Clump Thickness'] == 2)]
print("Sum: ",f1_2.shape[0])
# f1_2 = dataset[(dataset['Class'] == 2)]
# print("here")
print("Sum: ",f2_2.shape)
print("There")

# Create Cancer Likelihood Table
data_struct = [
        [cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Uniformity of Clump Thickness'] == 1)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Uniformity of Cell Size'] == 1)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Cell Shape'] == 1)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Marginal Adhesion'] == 1)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Single Epithelial Cell Size'] == 1)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Bare Nuclei'] == '1')].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Bland Chromatin'] == 1)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Normal Nucleoli'] == 1)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Mitoses'] == 1)].shape[0]],
        [cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Uniformity of Clump Thickness'] == 2)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Uniformity of Cell Size'] == 2)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Cell Shape'] == 2)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Marginal Adhesion'] == 2)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Single Epithelial Cell Size'] == 2)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Bare Nuclei'] == '2')].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Bland Chromatin'] == 2)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Normal Nucleoli'] == 2)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Mitoses'] == 2)].shape[0]],
        [cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Uniformity of Clump Thickness'] == 3)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Uniformity of Cell Size'] == 3)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Cell Shape'] == 3)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Marginal Adhesion'] == 3)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Single Epithelial Cell Size'] == 3)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Bare Nuclei'] == '3')].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Bland Chromatin'] == 3)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Normal Nucleoli'] == 3)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Mitoses'] == 3)].shape[0]],
        [cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Uniformity of Clump Thickness'] == 4)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Uniformity of Cell Size'] == 4)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Cell Shape'] == 4)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Marginal Adhesion'] == 4)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Single Epithelial Cell Size'] == 4)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Bare Nuclei'] == '4')].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Bland Chromatin'] == 4)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Normal Nucleoli'] == 4)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Mitoses'] == 4)].shape[0]],
        [cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Uniformity of Clump Thickness'] == 5)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Uniformity of Cell Size'] == 5)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Cell Shape'] == 5)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Marginal Adhesion'] == 5)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Single Epithelial Cell Size'] == 5)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Bare Nuclei'] == '5')].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Bland Chromatin'] == 5)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Normal Nucleoli'] == 5)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Mitoses'] == 5)].shape[0]],
        [cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Uniformity of Clump Thickness'] == 6)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Uniformity of Cell Size'] == 6)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Cell Shape'] == 6)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Marginal Adhesion'] == 6)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Single Epithelial Cell Size'] == 6)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Bare Nuclei'] == '6')].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Bland Chromatin'] == 6)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Normal Nucleoli'] == 6)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Mitoses'] == 6)].shape[0]],
        [cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Uniformity of Clump Thickness'] == 7)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Uniformity of Cell Size'] == 7)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Cell Shape'] == 7)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Marginal Adhesion'] == 7)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Single Epithelial Cell Size'] == 7)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Bare Nuclei'] == '7')].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Bland Chromatin'] == 7)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Normal Nucleoli'] == 7)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Mitoses'] == 7)].shape[0]],
        [cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Uniformity of Clump Thickness'] == 8)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Uniformity of Cell Size'] == 8)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Cell Shape'] == 8)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Marginal Adhesion'] == 8)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Single Epithelial Cell Size'] == 8)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Bare Nuclei'] == '8')].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Bland Chromatin'] == 8)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Normal Nucleoli'] == 8)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Mitoses'] == 8)].shape[0]],
        [cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Uniformity of Clump Thickness'] == 9)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Uniformity of Cell Size'] == 9)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Cell Shape'] == 9)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Marginal Adhesion'] == 9)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Single Epithelial Cell Size'] == 9)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Bare Nuclei'] == '9')].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Bland Chromatin'] == 9)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Normal Nucleoli'] == 9)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Mitoses'] == 9)].shape[0]],
        [cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Uniformity of Clump Thickness'] == 10)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Uniformity of Cell Size'] == 10)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Cell Shape'] == 10)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Marginal Adhesion'] == 10)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Single Epithelial Cell Size'] == 10)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Bare Nuclei'] == '10')].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Bland Chromatin'] == 10)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Normal Nucleoli'] == 10)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 2) & (cancer_dataset['Mitoses'] == 10)].shape[0]],
        [cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Uniformity of Clump Thickness'] == 1)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Uniformity of Cell Size'] == 1)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Cell Shape'] == 1)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Marginal Adhesion'] == 1)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Single Epithelial Cell Size'] == 1)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Bare Nuclei'] == '1')].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Bland Chromatin'] == 1)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Normal Nucleoli'] == 1)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Mitoses'] == 1)].shape[0]],
        [cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Uniformity of Clump Thickness'] == 2)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Uniformity of Cell Size'] == 2)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Cell Shape'] == 2)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Marginal Adhesion'] == 2)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Single Epithelial Cell Size'] == 2)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Bare Nuclei'] == '2')].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Bland Chromatin'] == 2)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Normal Nucleoli'] == 2)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Mitoses'] == 2)].shape[0]],
        [cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Uniformity of Clump Thickness'] == 3)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Uniformity of Cell Size'] == 3)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Cell Shape'] == 3)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Marginal Adhesion'] == 3)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Single Epithelial Cell Size'] == 3)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Bare Nuclei'] == '3')].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Bland Chromatin'] == 3)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Normal Nucleoli'] == 3)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Mitoses'] == 3)].shape[0]],
        [cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Uniformity of Clump Thickness'] == 4)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Uniformity of Cell Size'] == 4)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Cell Shape'] == 4)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Marginal Adhesion'] == 4)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Single Epithelial Cell Size'] == 4)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Bare Nuclei'] == '4')].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Bland Chromatin'] == 4)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Normal Nucleoli'] == 4)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Mitoses'] == 4)].shape[0]],
        [cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Uniformity of Clump Thickness'] == 5)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Uniformity of Cell Size'] == 5)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Cell Shape'] == 5)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Marginal Adhesion'] == 5)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Single Epithelial Cell Size'] == 5)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Bare Nuclei'] == '5')].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Bland Chromatin'] == 5)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Normal Nucleoli'] == 5)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Mitoses'] == 5)].shape[0]],
        [cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Uniformity of Clump Thickness'] == 6)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Uniformity of Cell Size'] == 6)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Cell Shape'] == 6)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Marginal Adhesion'] == 6)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Single Epithelial Cell Size'] == 6)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Bare Nuclei'] == '6')].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Bland Chromatin'] == 6)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Normal Nucleoli'] == 6)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Mitoses'] == 6)].shape[0]],
        [cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Uniformity of Clump Thickness'] == 7)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Uniformity of Cell Size'] == 7)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Cell Shape'] == 7)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Marginal Adhesion'] == 7)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Single Epithelial Cell Size'] == 7)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Bare Nuclei'] == '7')].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Bland Chromatin'] == 7)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Normal Nucleoli'] == 7)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Mitoses'] == 7)].shape[0]],
        [cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Uniformity of Clump Thickness'] == 8)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Uniformity of Cell Size'] == 8)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Cell Shape'] == 8)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Marginal Adhesion'] == 8)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Single Epithelial Cell Size'] == 8)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Bare Nuclei'] == '8')].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Bland Chromatin'] == 8)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Normal Nucleoli'] == 8)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Mitoses'] == 8)].shape[0]],
        [cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Uniformity of Clump Thickness'] == 9)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Uniformity of Cell Size'] == 9)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Cell Shape'] == 9)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Marginal Adhesion'] == 9)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Single Epithelial Cell Size'] == 9)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Bare Nuclei'] == '9')].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Bland Chromatin'] == 9)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Normal Nucleoli'] == 9)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Mitoses'] == 9)].shape[0]],
        [cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Uniformity of Clump Thickness'] == 10)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Uniformity of Cell Size'] == 10)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Cell Shape'] == 10)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Marginal Adhesion'] == 10)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Single Epithelial Cell Size'] == 10)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Bare Nuclei'] == '10')].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Bland Chromatin'] == 10)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Normal Nucleoli'] == 10)].shape[0],cancer_dataset[(cancer_dataset['Class'] == 4) & (cancer_dataset['Mitoses'] == 10)].shape[0]]]

cancer_frequency = pd.DataFrame(data_struct, columns=["Uniformity of Clump Thickness", "Uniformity of Cell Size", "Cell Shape", "Marginal Adhesion", "Single Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses"], index=['1-2','2-2','3-2','4-2','5-2','6-2','7-2','8-2','9-2','10-2','1-4','2-4','3-4','4-4','5-4','6-4','7-4','8-4','9-4','10-4'])

print(cancer_frequency)
print('Bare Nuclei')
print(f6_df)
# print(dataset['Bare Nuclei'])
# print('Bare Nuclei')
# print(dataset[(dataset['Class'] == 4) & (dataset['Bare Nuclei'] == '10')])
# print('Bare Nuclei')
# print(dataset[dataset['Bare Nuclei'] == 10])
# print('Bare Nuclei')
# print(dataset[dataset['Bare Nuclei'] == '10'])

num_features = len(cancer_frequency.columns)
print("---------------------")
print(num_features)

class_2_df = cancer_frequency.iloc[:10,:]
class_4_df = cancer_frequency.iloc[10:,:]
print(class_2_df)
print(class_4_df)

def calcualte_likehood(a_c, n_c, d):
    return (a_c + 1)/(n_c + d)


class_2_df = class_2_df.apply(calcualte_likehood, args=(class_two, num_features))
print(class_2_df)
class_4_df = class_4_df.apply(calcualte_likehood, args=(class_four, num_features))
print(class_4_df)

cancer_likelihood = class_2_df.append(class_4_df)
print(cancer_likelihood)


print("########### Naive Bayes Algorithm ##############")
def calculate_feature_product(test_set, train_set_likelihood_table, train_type_dict):
    for row_index, row in test_set.iterrows():
        f_product_2 = 1;
        f_product_4 = 1;
        for col_idx, value in row.items():
            if col_idx != "Sample code number" and col_idx != "Class" and col_idx != "class":
                for t_key in train_type_dict:
                    if t_key == 2:
                        label = str(value) + "-" + str(t_key)
                        f_product_2 *= train_set_likelihood_table.loc[label, col_idx]
                    if t_key == 4:
                        label = str(value) + "-" + str(t_key)
                        f_product_4 *= train_set_likelihood_table.loc[label, col_idx]
        c_2 = f_product_2 * train_type_dict[2]
        # print("c_2", c_2)
        c_4 = f_product_4 * train_type_dict[4]
        if c_2 > c_4:
            test_set["New_Class"] = 2
        else:
            test_set["New_Class"] = 4
        test_set.to_csv('classes.csv')
        # print("c_4", c_4)
        
    return test_set
        


print("############################# 10 fold Cross Validation ###########################")

# randomizing the data set
cancer_dataset = cancer_dataset.reindex(np.random.permutation(cancer_dataset.index))         
   

# reset the index
cancer_dataset = cancer_dataset.reset_index(drop=True)


# make 10 different fold with the same size.
fold1, fold2, fold3, fold4, fold5, fold6, fold7, fold8, fold9, fold10 = np.array_split(cancer_dataset, 10)
print(fold1)


### set up training sets
print("########### Train_Set_1 Likelihood Table ###################")
train_set1 = pd.concat([fold1, fold2, fold3, fold4, fold5, fold6, fold7, fold8, fold9])
test_set1 = fold10
data_struct = [
        [train_set1[(train_set1['Class'] == 2) & (train_set1['Uniformity of Clump Thickness'] == 1)].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Uniformity of Cell Size'] == 1)].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Cell Shape'] == 1)].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Marginal Adhesion'] == 1)].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Single Epithelial Cell Size'] == 1)].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Bare Nuclei'] == '1')].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Bland Chromatin'] == 1)].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Normal Nucleoli'] == 1)].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Mitoses'] == 1)].shape[0]],
        [train_set1[(train_set1['Class'] == 2) & (train_set1['Uniformity of Clump Thickness'] == 2)].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Uniformity of Cell Size'] == 2)].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Cell Shape'] == 2)].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Marginal Adhesion'] == 2)].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Single Epithelial Cell Size'] == 2)].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Bare Nuclei'] == '2')].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Bland Chromatin'] == 2)].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Normal Nucleoli'] == 2)].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Mitoses'] == 2)].shape[0]],
        [train_set1[(train_set1['Class'] == 2) & (train_set1['Uniformity of Clump Thickness'] == 3)].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Uniformity of Cell Size'] == 3)].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Cell Shape'] == 3)].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Marginal Adhesion'] == 3)].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Single Epithelial Cell Size'] == 3)].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Bare Nuclei'] == '3')].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Bland Chromatin'] == 3)].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Normal Nucleoli'] == 3)].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Mitoses'] == 3)].shape[0]],
        [train_set1[(train_set1['Class'] == 2) & (train_set1['Uniformity of Clump Thickness'] == 4)].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Uniformity of Cell Size'] == 4)].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Cell Shape'] == 4)].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Marginal Adhesion'] == 4)].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Single Epithelial Cell Size'] == 4)].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Bare Nuclei'] == '4')].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Bland Chromatin'] == 4)].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Normal Nucleoli'] == 4)].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Mitoses'] == 4)].shape[0]],
        [train_set1[(train_set1['Class'] == 2) & (train_set1['Uniformity of Clump Thickness'] == 5)].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Uniformity of Cell Size'] == 5)].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Cell Shape'] == 5)].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Marginal Adhesion'] == 5)].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Single Epithelial Cell Size'] == 5)].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Bare Nuclei'] == '5')].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Bland Chromatin'] == 5)].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Normal Nucleoli'] == 5)].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Mitoses'] == 5)].shape[0]],
        [train_set1[(train_set1['Class'] == 2) & (train_set1['Uniformity of Clump Thickness'] == 6)].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Uniformity of Cell Size'] == 6)].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Cell Shape'] == 6)].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Marginal Adhesion'] == 6)].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Single Epithelial Cell Size'] == 6)].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Bare Nuclei'] == '6')].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Bland Chromatin'] == 6)].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Normal Nucleoli'] == 6)].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Mitoses'] == 6)].shape[0]],
        [train_set1[(train_set1['Class'] == 2) & (train_set1['Uniformity of Clump Thickness'] == 7)].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Uniformity of Cell Size'] == 7)].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Cell Shape'] == 7)].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Marginal Adhesion'] == 7)].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Single Epithelial Cell Size'] == 7)].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Bare Nuclei'] == '7')].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Bland Chromatin'] == 7)].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Normal Nucleoli'] == 7)].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Mitoses'] == 7)].shape[0]],
        [train_set1[(train_set1['Class'] == 2) & (train_set1['Uniformity of Clump Thickness'] == 8)].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Uniformity of Cell Size'] == 8)].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Cell Shape'] == 8)].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Marginal Adhesion'] == 8)].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Single Epithelial Cell Size'] == 8)].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Bare Nuclei'] == '8')].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Bland Chromatin'] == 8)].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Normal Nucleoli'] == 8)].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Mitoses'] == 8)].shape[0]],
        [train_set1[(train_set1['Class'] == 2) & (train_set1['Uniformity of Clump Thickness'] == 9)].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Uniformity of Cell Size'] == 9)].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Cell Shape'] == 9)].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Marginal Adhesion'] == 9)].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Single Epithelial Cell Size'] == 9)].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Bare Nuclei'] == '9')].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Bland Chromatin'] == 9)].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Normal Nucleoli'] == 9)].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Mitoses'] == 9)].shape[0]],
        [train_set1[(train_set1['Class'] == 2) & (train_set1['Uniformity of Clump Thickness'] == 10)].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Uniformity of Cell Size'] == 10)].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Cell Shape'] == 10)].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Marginal Adhesion'] == 10)].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Single Epithelial Cell Size'] == 10)].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Bare Nuclei'] == '10')].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Bland Chromatin'] == 10)].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Normal Nucleoli'] == 10)].shape[0],train_set1[(train_set1['Class'] == 2) & (train_set1['Mitoses'] == 10)].shape[0]],
        [train_set1[(train_set1['Class'] == 4) & (train_set1['Uniformity of Clump Thickness'] == 1)].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Uniformity of Cell Size'] == 1)].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Cell Shape'] == 1)].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Marginal Adhesion'] == 1)].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Single Epithelial Cell Size'] == 1)].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Bare Nuclei'] == '1')].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Bland Chromatin'] == 1)].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Normal Nucleoli'] == 1)].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Mitoses'] == 1)].shape[0]],
        [train_set1[(train_set1['Class'] == 4) & (train_set1['Uniformity of Clump Thickness'] == 2)].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Uniformity of Cell Size'] == 2)].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Cell Shape'] == 2)].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Marginal Adhesion'] == 2)].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Single Epithelial Cell Size'] == 2)].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Bare Nuclei'] == '2')].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Bland Chromatin'] == 2)].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Normal Nucleoli'] == 2)].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Mitoses'] == 2)].shape[0]],
        [train_set1[(train_set1['Class'] == 4) & (train_set1['Uniformity of Clump Thickness'] == 3)].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Uniformity of Cell Size'] == 3)].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Cell Shape'] == 3)].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Marginal Adhesion'] == 3)].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Single Epithelial Cell Size'] == 3)].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Bare Nuclei'] == '3')].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Bland Chromatin'] == 3)].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Normal Nucleoli'] == 3)].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Mitoses'] == 3)].shape[0]],
        [train_set1[(train_set1['Class'] == 4) & (train_set1['Uniformity of Clump Thickness'] == 4)].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Uniformity of Cell Size'] == 4)].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Cell Shape'] == 4)].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Marginal Adhesion'] == 4)].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Single Epithelial Cell Size'] == 4)].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Bare Nuclei'] == '4')].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Bland Chromatin'] == 4)].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Normal Nucleoli'] == 4)].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Mitoses'] == 4)].shape[0]],
        [train_set1[(train_set1['Class'] == 4) & (train_set1['Uniformity of Clump Thickness'] == 5)].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Uniformity of Cell Size'] == 5)].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Cell Shape'] == 5)].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Marginal Adhesion'] == 5)].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Single Epithelial Cell Size'] == 5)].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Bare Nuclei'] == '5')].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Bland Chromatin'] == 5)].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Normal Nucleoli'] == 5)].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Mitoses'] == 5)].shape[0]],
        [train_set1[(train_set1['Class'] == 4) & (train_set1['Uniformity of Clump Thickness'] == 6)].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Uniformity of Cell Size'] == 6)].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Cell Shape'] == 6)].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Marginal Adhesion'] == 6)].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Single Epithelial Cell Size'] == 6)].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Bare Nuclei'] == '6')].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Bland Chromatin'] == 6)].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Normal Nucleoli'] == 6)].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Mitoses'] == 6)].shape[0]],
        [train_set1[(train_set1['Class'] == 4) & (train_set1['Uniformity of Clump Thickness'] == 7)].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Uniformity of Cell Size'] == 7)].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Cell Shape'] == 7)].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Marginal Adhesion'] == 7)].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Single Epithelial Cell Size'] == 7)].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Bare Nuclei'] == '7')].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Bland Chromatin'] == 7)].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Normal Nucleoli'] == 7)].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Mitoses'] == 7)].shape[0]],
        [train_set1[(train_set1['Class'] == 4) & (train_set1['Uniformity of Clump Thickness'] == 8)].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Uniformity of Cell Size'] == 8)].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Cell Shape'] == 8)].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Marginal Adhesion'] == 8)].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Single Epithelial Cell Size'] == 8)].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Bare Nuclei'] == '8')].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Bland Chromatin'] == 8)].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Normal Nucleoli'] == 8)].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Mitoses'] == 8)].shape[0]],
        [train_set1[(train_set1['Class'] == 4) & (train_set1['Uniformity of Clump Thickness'] == 9)].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Uniformity of Cell Size'] == 9)].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Cell Shape'] == 9)].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Marginal Adhesion'] == 9)].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Single Epithelial Cell Size'] == 9)].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Bare Nuclei'] == '9')].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Bland Chromatin'] == 9)].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Normal Nucleoli'] == 9)].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Mitoses'] == 9)].shape[0]],
        [train_set1[(train_set1['Class'] == 4) & (train_set1['Uniformity of Clump Thickness'] == 10)].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Uniformity of Cell Size'] == 10)].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Cell Shape'] == 10)].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Marginal Adhesion'] == 10)].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Single Epithelial Cell Size'] == 10)].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Bare Nuclei'] == '10')].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Bland Chromatin'] == 10)].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Normal Nucleoli'] == 10)].shape[0],train_set1[(train_set1['Class'] == 4) & (train_set1['Mitoses'] == 10)].shape[0]]]

# CORRECT!!!
####################
train_set1_last_n_column  = train_set1.iloc[: , -1:]
train_set1_N = len(train_set1_last_n_column)

train_set1_class_two = train_set1_last_n_column.value_counts()[2]
train_set1_class_four = train_set1_last_n_column.value_counts()[4]
train_set1_class_two_prob = train_set1_class_two / train_set1_N
train_set1_class_four_prob = train_set1_class_four / train_set1_N
class_type_dict = {2:train_set1_class_two_prob, 4:train_set1_class_four_prob}
print(train_set1_N,",",train_set1_class_two,",",train_set1_class_four,",",train_set1_class_two_prob,",",train_set1_class_four_prob)
print(class_type_dict)
#############################

# CORRECT!!!
#############################
train_set1_frequency_table = pd.DataFrame(data_struct, columns=["Uniformity of Clump Thickness", "Uniformity of Cell Size", "Cell Shape", "Marginal Adhesion", "Single Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses"], index=['1-2','2-2','3-2','4-2','5-2','6-2','7-2','8-2','9-2','10-2','1-4','2-4','3-4','4-4','5-4','6-4','7-4','8-4','9-4','10-4'])
print(train_set1_frequency_table)
train_set1_num_features = len(train_set1_frequency_table.columns)
print(train_set1_num_features)
train_set1_class_2_df = train_set1_frequency_table.iloc[:10,:]
print(train_set1_class_2_df)
print("the total counts of a dataframe ****************************************")
sum2 = train_set1_class_2_df.sum()
total_number_feature_in_class_2 = sum2.sum()
print(total_number_feature_in_class_2)

train_set1_class_4_df = train_set1_frequency_table.iloc[10:,:]
print(train_set1_class_4_df)
sum4= train_set1_class_4_df.sum()
total_number_feature_in_class_4 = sum4.sum()
print("the total counts of a dataframe ****************************************")
print(total_number_feature_in_class_4)

#######################################

#CORRECT?
############################## Training function ######################
train_set1_class_2_df = train_set1_class_2_df.apply(calcualte_likehood, args=(total_number_feature_in_class_2, train_set1_num_features))
train_set1_class_4_df = train_set1_class_4_df.apply(calcualte_likehood, args=(total_number_feature_in_class_4, train_set1_num_features))
train_set1_likelihood = train_set1_class_2_df.append(train_set1_class_4_df)

print(train_set1_class_2_df)
print(train_set1_likelihood)

#CORRECT?
############################## Testing function #######################
print("##############navie bayes result")
#new_test_set_1= calculate_feature_product(test_set1, train_set1_likelihood, class_type_dict)
print(".......................")
#print(new_test_set_1)



print("########### Train set 2 ###################")

train_set2 = pd.concat([fold1, fold2, fold3, fold4, fold5, fold6, fold7, fold8, fold10])
test_set2 = fold9
print(train_set2)


# CORRECT!!!
####################
train_set2_last_n_column  = train_set2.iloc[: , -1:]
train_set2_N = len(train_set2_last_n_column)
print(train_set2_last_n_column)


#############################




train_set3 =  pd.concat([fold1, fold2, fold3, fold4, fold5, fold6, fold7, fold9, fold10])
test_set3 = fold8

train_set4 =  pd.concat([fold1, fold2, fold3, fold4, fold5, fold6, fold8, fold9, fold10])
test_set4 = fold7

train_set5 =  pd.concat([fold1, fold2, fold3, fold4, fold5, fold7, fold8, fold9, fold10])
test_set5 = fold6

train_set6 =  pd.concat([fold1, fold2, fold3, fold4, fold6, fold7, fold8, fold9, fold10])
test_set6 = fold5

train_set7 =  pd.concat([fold1, fold2, fold3, fold5, fold6, fold7, fold8, fold9, fold10])
test_set67= fold4

train_set8 =  pd.concat([fold1, fold2, fold4, fold5, fold6, fold7, fold8, fold9, fold10])
test_set8 = fold3

train_set9 =  pd.concat([fold1, fold3, fold4, fold5, fold6, fold7, fold8, fold9, fold10])
test_set9 = fold2

train_set10 =  pd.concat([fold2, fold3, fold4, fold5, fold6, fold7, fold8, fold9, fold10])
test_set10 = fold1






  

            
            