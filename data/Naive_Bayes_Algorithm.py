from ctypes import string_at
import numpy as np
import pandas as pd
# clean our data, get rid of ? 
def clean_data(dataset):
    dataset = dataset.replace("?", np.NaN)
    dataset = dataset.dropna()
    return dataset

def stratify(dataset):
    # get all attributes that has class label
    print("###########Entering Stratify Function#################")
    folds = [[] for x in range(10)]
    # class as key and all attributes as values
    class_dataset = {}
    class_names = dataset["Class"]
    class_names = class_names.unique()
    for name in class_names:
        class_dataset[name] = dataset[dataset["Class"] == name]
        print( class_dataset[name])

    for key in class_dataset:
        class_count = class_dataset[key]["Class"].count()
        fold_index = int (class_count / 10)
        class_list = class_dataset[key].values.tolist()
        for i in range(10):
            start = int(i * fold_index)
            end = int(start + fold_index)
            folds[i] += class_list[start:end]
    return class_dataset, folds

def calculate_class_probability(class_dataset):
    class_probability = {}
    total = 0;
    for key, value in class_dataset.items():
        total += value["Class"].count()
    for key, value in class_dataset.items():
        class_probability[key] = value["Class"].count() / total
    return class_probability
 

    
# def calculate_frequency_table(dataset):
#     dataset = dataset.iloc[1: , :]
#     print(dataset)
#     attribute_frequency_table = {}
#     for i in range(dataset.shape[0]):
#         for j in range(dataset.shape[1]):
#             class_name = dataset.iloc[i][10]
#             if j != 10:
#                 col = j
#                 val = dataset.iloc[i][j]
#                 try: #if we have seen this attribute value before, add 1 to the count
#                     att_count = attribute_frequency_table[class_name][col][val]
#                     attribute_frequency_table[class_name][col][val] = att_count + 1
#                     #attribute_counts[class_name][attribute_value]+=1 #simplification of last two lines
#                 except KeyError:
#                     try:#if we have not seen this attribute value before, make the count 1
#                         attribute_frequency_table[class_name][col].update({val:1})
#                     except KeyError:
#                         try:#if we have not seen this attribute column (ie attribute) before initialize the attirbute
#                             attribute_frequency_table[class_name].update({col:{val:1}})
#                         except KeyError: #if we have not seen this class before initialize the class
#                             attribute_frequency_table.update({class_name:{col:{val:1}}})
#     return attribute_frequency_table

# def calculate_likelihood_table(attribute_frequency_table, class_frequency):
#     attribute_likelihood_table = {}
#     for key in attribute_frequency_table:
#         for col in attribute_frequency_table[key]:
#             for attr in attribute_frequency_table[key][col]:
#                 attribute_count = attribute_frequency_table[key][col][attr] + 1  #laplace smoother here
#                 conditional_probability = attribute_count / (class_frequency[key] + 9)#prob = occurances of attribute given a class / total occurences of given class
#                 attribute_likelihood_table.update({key:{col:{attr:conditional_probability}}})
#     return attribute_likelihood_table

def main():
    file_name = "cancer.data"
    dataset = pd.read_csv(file_name, sep=",")
    dataset = clean_data(dataset)
    dataset.columns = ["Sample code number", "Uniformity of Clump Thickness", "Uniformity of Cell Size", "Cell Shape", "Marginal Adhesion", "Single Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses", "Class"]
    print(dataset)
    dataset = dataset.sort_values(by=['Class'])
    class_dataset, folds = stratify(dataset)
    class_probablity = calculate_class_probability(class_dataset)
    print(class_probablity)
    
    # class_attributes, class_frequency, folds = stratify(dataset)
    # class_probability = calculate_class_probability(class_frequency)
    # print(class_probability)
    
    # attribute_frequency_table = calculate_frequency_table(dataset)
    # attribute_likelihood_table = calculate_likelihood_table(attribute_frequency_table, class_frequency)
    # print(attribute_likelihood_table)
    
main()
