from ast import Delete
from ctypes import sizeof
from os import remove
import pandas as pd
import numpy as np
import scipy as sp

### read in the file
file_name = "breast-cancer-wisconsin.data"
dataset = pd.read_csv(file_name, sep=",")

# drop all the ? cells
dataset = dataset.replace("?", np.NaN)
dataset = dataset.dropna()

# sorted data by class
dataset.columns = ["Sample code number", "Uniformity of Clump Thickness", "Uniformity of Cell Size", "Cell Shape", "Marginal Adhesion", "Single Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses", "Class"]
dataset = dataset.sort_values(by=['Class'])

print(dataset)


#### likelihood
def calcualte_likehood(a_c, n_c, d):
    return (a_c + 1)/(n_c + d)

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
                      
        print("f_product_2", f_product_2)
        print("f_product_4", f_product_4)
        c_2 = f_product_2 * train_type_dict[2]
        c_4 = f_product_4 * train_type_dict[4]
        print("c_2", c_2)
        print("c_4", c_4)
        if c_2 > c_4:
            test_set["Classifier"] = 2
        else:
            test_set["Classifier"] = 4
        test_set.to_csv('classes.csv')
    return test_set


print("############################# 10 fold Cross Validation ###########################")
dataset_2 = dataset[dataset["Class"] == 2]
dataset_4 = dataset[dataset["Class"] == 4]
# randomizing the data set
dataset = dataset.reindex(np.random.permutation(dataset.index))         
# reset the index
dataset = dataset.reset_index(drop=True)
# make 10 different fold with the same size.
#fold1, fold2, fold3, fold4, fold5, fold6, fold7, fold8, fold9, fold10 = np.array_split(dataset, 10)
fold1_2, fold2_2, fold3_2, fold4_2, fold5_2, fold6_2, fold7_2, fold8_2, fold9_2, fold10_2 = np.array_split(dataset_2, 10)
fold1_4, fold2_4, fold3_4, fold4_4, fold5_4, fold6_4, fold7_4, fold8_4, fold9_4, fold10_4 = np.array_split(dataset_4, 10)

fold1 = pd.concat([fold1_2, fold1_4])
fold2 = pd.concat([fold2_2, fold2_4])
fold3 = pd.concat([fold3_2, fold3_4])
fold4 = pd.concat([fold4_2, fold4_4])
fold5 = pd.concat([fold5_2, fold5_4])
fold6 = pd.concat([fold6_2, fold6_4])
fold7 = pd.concat([fold7_2, fold7_4])
fold8 = pd.concat([fold8_2, fold8_4])
fold9 = pd.concat([fold9_2, fold9_4])
fold10 = pd.concat([fold10_2, fold10_4])
print("hahhhhhhhhhhhhhh")
print(fold1)

for each in [fold1,fold2,fold3,fold4,fold5,fold6,fold7,fold8,fold9,fold10]:
    print(len(each[each['Class'] == 2]))
    print(len(each[each['Class'] == 4]))
### set up training sets
print("########### Train_Set_1 ###################")
train_set1 = pd.concat([fold1, fold2, fold3, fold4, fold5, fold6, fold7, fold8, fold9])
print(train_set1)
test_set1 = fold10.copy()
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

train_set1_frequency_table = pd.DataFrame(data_struct, columns=["Uniformity of Clump Thickness", "Uniformity of Cell Size", "Cell Shape", "Marginal Adhesion", "Single Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses"], index=['1-2','2-2','3-2','4-2','5-2','6-2','7-2','8-2','9-2','10-2','1-4','2-4','3-4','4-4','5-4','6-4','7-4','8-4','9-4','10-4'])
print(train_set1_frequency_table)


train_set1_last_n_column  = train_set1.iloc[: , -1:]
train_set1_N = len(train_set1_last_n_column)
train_set1_class_two = train_set1_last_n_column.value_counts()[2]
train_set1_class_four = train_set1_last_n_column.value_counts()[4]
train_set1_class_two_prob = train_set1_class_two / train_set1_N
train_set1_class_four_prob = train_set1_class_four / train_set1_N
class_type_dict = {2:train_set1_class_two_prob, 4:train_set1_class_four_prob}
print("total number of samples: " , train_set1_N, "\n")
print("total number of class two samples: ", train_set1_class_two, "\n")
print("total number of class four samples: ", train_set1_class_four, "\n")
print("probability of class two: ", train_set1_class_two_prob, "\n")
print("probability of class four: ", train_set1_class_four_prob, "\n")
print(class_type_dict, "\n")

train_set1_num_features = len(train_set1_frequency_table.columns)
print("train_set1_num_features: ", train_set1_num_features, "\n")

train_set1_class_2_freq_table = train_set1_frequency_table.iloc[:10,:]
train_set1_class_4_freq_table = train_set1_frequency_table.iloc[10:,:]
train_set1_class_2_freq_table.to_csv("train_set1_class_2_freq_table.csv")
### apply likelihood
train_set1_class_2_likelihood_table = train_set1_class_2_freq_table.apply(calcualte_likehood, args=(train_set1_class_two, train_set1_num_features))
train_set1_class_4_likelihood_table = train_set1_class_4_freq_table.apply(calcualte_likehood, args=(train_set1_class_four, train_set1_num_features))
train_set1_likelihood = pd.concat([train_set1_class_2_likelihood_table, train_set1_class_4_likelihood_table])

print("train_set1_likelihood", train_set1_likelihood)

train_set1_likelihood.to_csv("train_set1_likelihood.csv")


### apply navie bayes algorithm 
test_set1 = calculate_feature_product(test_set1, train_set1_likelihood, class_type_dict)
print("############RESULT##############################")
print(test_set1)

print("########### Train set 2 ###################")
train_set2 = pd.concat([fold1, fold2, fold3, fold4, fold5, fold6, fold7, fold8, fold10])
# i=1
# for row in [fold1, fold2, fold3, fold4,fold5,fold6,fold7,fold8,fold9,fold10]:
#     print(i)
#     print(row)
#     i+=1
# print("YAHOIJSDOHKLFJKSDF>J")
# print(train_set2)
test_set2 = fold9

data_struct = [
        [train_set2[(train_set2['Class'] == 2) & (train_set2['Uniformity of Clump Thickness'] == 1)].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Uniformity of Cell Size'] == 1)].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Cell Shape'] == 1)].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Marginal Adhesion'] == 1)].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Single Epithelial Cell Size'] == 1)].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Bare Nuclei'] == '1')].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Bland Chromatin'] == 1)].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Normal Nucleoli'] == 1)].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Mitoses'] == 1)].shape[0]],
        [train_set2[(train_set2['Class'] == 2) & (train_set2['Uniformity of Clump Thickness'] == 2)].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Uniformity of Cell Size'] == 2)].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Cell Shape'] == 2)].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Marginal Adhesion'] == 2)].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Single Epithelial Cell Size'] == 2)].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Bare Nuclei'] == '2')].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Bland Chromatin'] == 2)].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Normal Nucleoli'] == 2)].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Mitoses'] == 2)].shape[0]],
        [train_set2[(train_set2['Class'] == 2) & (train_set2['Uniformity of Clump Thickness'] == 3)].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Uniformity of Cell Size'] == 3)].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Cell Shape'] == 3)].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Marginal Adhesion'] == 3)].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Single Epithelial Cell Size'] == 3)].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Bare Nuclei'] == '3')].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Bland Chromatin'] == 3)].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Normal Nucleoli'] == 3)].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Mitoses'] == 3)].shape[0]],
        [train_set2[(train_set2['Class'] == 2) & (train_set2['Uniformity of Clump Thickness'] == 4)].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Uniformity of Cell Size'] == 4)].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Cell Shape'] == 4)].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Marginal Adhesion'] == 4)].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Single Epithelial Cell Size'] == 4)].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Bare Nuclei'] == '4')].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Bland Chromatin'] == 4)].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Normal Nucleoli'] == 4)].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Mitoses'] == 4)].shape[0]],
        [train_set2[(train_set2['Class'] == 2) & (train_set2['Uniformity of Clump Thickness'] == 5)].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Uniformity of Cell Size'] == 5)].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Cell Shape'] == 5)].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Marginal Adhesion'] == 5)].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Single Epithelial Cell Size'] == 5)].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Bare Nuclei'] == '5')].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Bland Chromatin'] == 5)].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Normal Nucleoli'] == 5)].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Mitoses'] == 5)].shape[0]],
        [train_set2[(train_set2['Class'] == 2) & (train_set2['Uniformity of Clump Thickness'] == 6)].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Uniformity of Cell Size'] == 6)].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Cell Shape'] == 6)].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Marginal Adhesion'] == 6)].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Single Epithelial Cell Size'] == 6)].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Bare Nuclei'] == '6')].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Bland Chromatin'] == 6)].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Normal Nucleoli'] == 6)].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Mitoses'] == 6)].shape[0]],
        [train_set2[(train_set2['Class'] == 2) & (train_set2['Uniformity of Clump Thickness'] == 7)].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Uniformity of Cell Size'] == 7)].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Cell Shape'] == 7)].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Marginal Adhesion'] == 7)].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Single Epithelial Cell Size'] == 7)].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Bare Nuclei'] == '7')].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Bland Chromatin'] == 7)].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Normal Nucleoli'] == 7)].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Mitoses'] == 7)].shape[0]],
        [train_set2[(train_set2['Class'] == 2) & (train_set2['Uniformity of Clump Thickness'] == 8)].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Uniformity of Cell Size'] == 8)].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Cell Shape'] == 8)].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Marginal Adhesion'] == 8)].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Single Epithelial Cell Size'] == 8)].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Bare Nuclei'] == '8')].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Bland Chromatin'] == 8)].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Normal Nucleoli'] == 8)].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Mitoses'] == 8)].shape[0]],
        [train_set2[(train_set2['Class'] == 2) & (train_set2['Uniformity of Clump Thickness'] == 9)].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Uniformity of Cell Size'] == 9)].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Cell Shape'] == 9)].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Marginal Adhesion'] == 9)].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Single Epithelial Cell Size'] == 9)].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Bare Nuclei'] == '9')].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Bland Chromatin'] == 9)].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Normal Nucleoli'] == 9)].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Mitoses'] == 9)].shape[0]],
        [train_set2[(train_set2['Class'] == 2) & (train_set2['Uniformity of Clump Thickness'] == 10)].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Uniformity of Cell Size'] == 10)].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Cell Shape'] == 10)].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Marginal Adhesion'] == 10)].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Single Epithelial Cell Size'] == 10)].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Bare Nuclei'] == '10')].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Bland Chromatin'] == 10)].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Normal Nucleoli'] == 10)].shape[0],train_set2[(train_set2['Class'] == 2) & (train_set2['Mitoses'] == 10)].shape[0]],
        [train_set2[(train_set2['Class'] == 4) & (train_set2['Uniformity of Clump Thickness'] == 1)].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Uniformity of Cell Size'] == 1)].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Cell Shape'] == 1)].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Marginal Adhesion'] == 1)].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Single Epithelial Cell Size'] == 1)].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Bare Nuclei'] == '1')].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Bland Chromatin'] == 1)].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Normal Nucleoli'] == 1)].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Mitoses'] == 1)].shape[0]],
        [train_set2[(train_set2['Class'] == 4) & (train_set2['Uniformity of Clump Thickness'] == 2)].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Uniformity of Cell Size'] == 2)].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Cell Shape'] == 2)].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Marginal Adhesion'] == 2)].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Single Epithelial Cell Size'] == 2)].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Bare Nuclei'] == '2')].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Bland Chromatin'] == 2)].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Normal Nucleoli'] == 2)].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Mitoses'] == 2)].shape[0]],
        [train_set2[(train_set2['Class'] == 4) & (train_set2['Uniformity of Clump Thickness'] == 3)].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Uniformity of Cell Size'] == 3)].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Cell Shape'] == 3)].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Marginal Adhesion'] == 3)].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Single Epithelial Cell Size'] == 3)].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Bare Nuclei'] == '3')].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Bland Chromatin'] == 3)].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Normal Nucleoli'] == 3)].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Mitoses'] == 3)].shape[0]],
        [train_set2[(train_set2['Class'] == 4) & (train_set2['Uniformity of Clump Thickness'] == 4)].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Uniformity of Cell Size'] == 4)].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Cell Shape'] == 4)].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Marginal Adhesion'] == 4)].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Single Epithelial Cell Size'] == 4)].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Bare Nuclei'] == '4')].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Bland Chromatin'] == 4)].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Normal Nucleoli'] == 4)].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Mitoses'] == 4)].shape[0]],
        [train_set2[(train_set2['Class'] == 4) & (train_set2['Uniformity of Clump Thickness'] == 5)].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Uniformity of Cell Size'] == 5)].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Cell Shape'] == 5)].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Marginal Adhesion'] == 5)].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Single Epithelial Cell Size'] == 5)].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Bare Nuclei'] == '5')].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Bland Chromatin'] == 5)].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Normal Nucleoli'] == 5)].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Mitoses'] == 5)].shape[0]],
        [train_set2[(train_set2['Class'] == 4) & (train_set2['Uniformity of Clump Thickness'] == 6)].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Uniformity of Cell Size'] == 6)].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Cell Shape'] == 6)].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Marginal Adhesion'] == 6)].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Single Epithelial Cell Size'] == 6)].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Bare Nuclei'] == '6')].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Bland Chromatin'] == 6)].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Normal Nucleoli'] == 6)].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Mitoses'] == 6)].shape[0]],
        [train_set2[(train_set2['Class'] == 4) & (train_set2['Uniformity of Clump Thickness'] == 7)].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Uniformity of Cell Size'] == 7)].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Cell Shape'] == 7)].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Marginal Adhesion'] == 7)].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Single Epithelial Cell Size'] == 7)].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Bare Nuclei'] == '7')].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Bland Chromatin'] == 7)].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Normal Nucleoli'] == 7)].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Mitoses'] == 7)].shape[0]],
        [train_set2[(train_set2['Class'] == 4) & (train_set2['Uniformity of Clump Thickness'] == 8)].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Uniformity of Cell Size'] == 8)].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Cell Shape'] == 8)].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Marginal Adhesion'] == 8)].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Single Epithelial Cell Size'] == 8)].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Bare Nuclei'] == '8')].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Bland Chromatin'] == 8)].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Normal Nucleoli'] == 8)].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Mitoses'] == 8)].shape[0]],
        [train_set2[(train_set2['Class'] == 4) & (train_set2['Uniformity of Clump Thickness'] == 9)].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Uniformity of Cell Size'] == 9)].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Cell Shape'] == 9)].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Marginal Adhesion'] == 9)].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Single Epithelial Cell Size'] == 9)].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Bare Nuclei'] == '9')].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Bland Chromatin'] == 9)].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Normal Nucleoli'] == 9)].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Mitoses'] == 9)].shape[0]],
        [train_set2[(train_set2['Class'] == 4) & (train_set2['Uniformity of Clump Thickness'] == 10)].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Uniformity of Cell Size'] == 10)].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Cell Shape'] == 10)].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Marginal Adhesion'] == 10)].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Single Epithelial Cell Size'] == 10)].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Bare Nuclei'] == '10')].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Bland Chromatin'] == 10)].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Normal Nucleoli'] == 10)].shape[0],train_set2[(train_set2['Class'] == 4) & (train_set2['Mitoses'] == 10)].shape[0]]]

train_set2_frequency_table = pd.DataFrame(data_struct, columns=["Uniformity of Clump Thickness", "Uniformity of Cell Size", "Cell Shape", "Marginal Adhesion", "Single Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses"], index=['1-2','2-2','3-2','4-2','5-2','6-2','7-2','8-2','9-2','10-2','1-4','2-4','3-4','4-4','5-4','6-4','7-4','8-4','9-4','10-4'])
print(train_set2_frequency_table)

print(train_set2)
train_set2_last_n_column  = train_set2.iloc[: , -1:]
train_set2_N = len(train_set2_last_n_column)
train_set2_class_two = train_set2_last_n_column.value_counts()[2]
train_set2_class_four = train_set2_last_n_column.value_counts()[4]
train_set2_class_two_prob = train_set2_class_two / train_set2_N
train_set2_class_four_prob = train_set2_class_four / train_set2_N
class_type_dict = {2:train_set2_class_two_prob, 4:train_set2_class_four_prob}
print("total number of samples: " , train_set2_N, "\n")
print("total number of class two samples: ", train_set2_class_two, "\n")
print("total number of class four samples: ", train_set2_class_four, "\n")
print("probability of class two: ", train_set2_class_two_prob, "\n")
print("probability of class four: ", train_set2_class_four_prob, "\n")
print(class_type_dict, "\n")

train_set2_num_features = len(train_set2_frequency_table.columns)
print("train_set2_num_features: ", train_set2_num_features, "\n")

train_set2_class_2_freq_table = train_set2_frequency_table.iloc[:10,:]
sum_temp = train_set2_class_2_freq_table.sum()
train_set2_class_2_total_features= sum_temp.sum()
print("train_set2_class_2_total_features: ", train_set2_class_2_total_features)

train_set2_class_4_freq_table = train_set2_frequency_table.iloc[10:,:]
sum_temp = train_set2_class_4_freq_table.sum()
train_set2_class_4_total_features= sum_temp.sum()

### apply likelihood
train_set2_class_2_likelihood_table = train_set2_class_2_freq_table.apply(calcualte_likehood, args=(train_set2_class_two, train_set2_num_features))
train_set2_class_4_likelihood_table = train_set2_class_4_freq_table.apply(calcualte_likehood, args=(train_set2_class_four, train_set2_num_features))
train_set2_likelihood = pd.concat([train_set2_class_2_likelihood_table, train_set2_class_4_likelihood_table])
print("train_set2_likelihood", train_set2_likelihood)

### apply navie bayes algorithm 
test_set2 = calculate_feature_product(test_set2, train_set2_likelihood, class_type_dict)
print("############RESULT##############################")
print(test_set2)




