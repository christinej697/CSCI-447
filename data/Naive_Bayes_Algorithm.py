from contextlib import nullcontext
from ctypes import string_at
from pickle import NONE
import numpy as np
import pandas as pd
import random
# clean our data, get rid of ? 
def clean_data(dataset):
    dataset = dataset.replace("?", np.NaN)
    dataset = dataset.dropna()
    return dataset

# def getClassCounts(dataset, training_set, last_col):
#     class_counts = {}
#     class_names = dataset["Class"]
#     class_names = class_names.unique()
#     for name in class_names:
#         name_count = 0
#         for lst in training_set:
#             for col in lst:
#                 if col[last_col] == name:
#                     name_count += 1
#         class_counts[name] = name_count
#     return class_counts
def getClassCounts(class_data, folds):
    class_names = class_data.keys()
    class_counts = {}
    for name in class_names: 
        total_count = 0   
        for fold in folds:
            total_count += fold["Class"].value_counts()[name]
        class_counts[name] = total_count
    return class_counts

# def stratify(dataset):
#     # get all attributes that has class label
#     print("###########Entering Stratify Function#################")
#     # class as key and all attributes as values
#     class_dataset = {}
#     folds = [[] for x in range(10)]
#     class_names = dataset["Class"]
#     class_names = class_names.unique()
#     for name in class_names:
#         class_dataset[name] = dataset[dataset["Class"] == name]
    
#     for key in class_dataset:
#         class_count = class_dataset[key]["Class"].count()
#         fold_index = int (class_count / 10)
#         class_list = class_dataset[key].values.tolist()
#         for i in range(10):
#             start = int(i * fold_index)
#             end = int(start + fold_index)
#             folds[i] += class_list[start:end]
#     return class_dataset, folds

def stratify_with_df(dataset):
    # get all attributes that has class label
    print("###########Entering Stratify Function#################")
    # class as key and all attributes as values
    class_dataset = {}
    folds_data = []
    folds = [[] for x in range(10)]
    class_names = dataset["Class"]
    class_names = class_names.unique()
    for name in class_names:
        class_dataset[name] = dataset[dataset["Class"] == name]
    
    for key in class_dataset:
        folds = np.array_split(class_dataset[key], 10)
        folds_data.append(folds)

    answer = []
    for column in range(len(folds_data[0])):
        t = pd.DataFrame()
        for row in folds_data:
            t = pd.concat([t, row[column]])
        answer.append(t)
    return class_dataset, answer


def calculate_class_probability(class_counts):
    class_probability = {}
    total = 0;
    for key, val in class_counts.items():
        total += val
    for key, val in class_counts.items():
        class_probability[key] = val / total
    return class_probability
    
def calculate_frequency_table(trainset):
    dataset = pd.DataFrame()
    for fold in trainset:
            dataset = pd.concat([dataset, fold])
    attribute_frequency_table = {}
    for i in range(dataset.shape[0]):
        for j in range(dataset.shape[1]):
            class_name = dataset.iloc[i][10]
            if j != 10:
                col = j
                val = dataset.iloc[i][j]
                try:
                    att_count = attribute_frequency_table[class_name][col][val]
                    attribute_frequency_table[class_name][col][val] = att_count + 1
                except KeyError:
                    try:
                        attribute_frequency_table[class_name][col].update({val:1})
                    except KeyError:
                        try:
                            attribute_frequency_table[class_name].update({col:{val:1}})
                        except KeyError:
                            attribute_frequency_table.update({class_name:{col:{val:1}}})
    return attribute_frequency_table
    
def calculate_likelihood_table(attribute_frequence_table, class_counts):
    for key in attribute_frequence_table:
        for col in attribute_frequence_table[key]:
            for attr in attribute_frequence_table[key][col]:
                attribute_count = attribute_frequence_table[key][col][attr] + 1  
                conditional_probability = attribute_count / (class_counts[key] + 9)
                attribute_frequence_table[key][col][attr] = conditional_probability
    return attribute_frequence_table

    
def naive_bayes_classify(testdata, attribute_likelihood_table, class_probablity):
    class_prob ={}
    for class_name in attribute_likelihood_table:
        c_prob = class_probablity[class_name]
        feature_product = 1
        for col in testdata:
            if (col != "Sample code number" and col != "Class"):
                for col_index in testdata[col]:
                        attr_value = col_index
                        if feature_product != 0:
                            feature_product *= attribute_likelihood_table[class_name][col_index][attr_value]
                        else:
                            feature_product = 1/10
                c_class = c_prob * feature_product
                class_prob.update({c_class:class_prob})
    print(class_prob)

def getData(test_data):
    dataset = pd.DataFrame()
    for fold in test_data:
            dataset = pd.concat([dataset, fold])
    return dataset

def cross_validation(class_data, folds):
    training = random.sample(folds, 9)
    testdata = random.sample(folds, 1)
    class_counts = getClassCounts(class_data, training)
    class_probs = calculate_class_probability(class_counts)
    freq_table = calculate_frequency_table(training)
    classAttributelikelihood = calculate_likelihood_table(freq_table, class_counts)
    test_set = getData(testdata)
    guess = naive_bayes_classify(test_set, classAttributelikelihood, class_probs)
    print(guess)
    
            
    
def main():
    file_name = "cancer.data"
    dataset = pd.read_csv(file_name, sep=",")
    dataset = clean_data(dataset)
    dataset.columns = ["Sample code number", "Uniformity of Clump Thickness", "Uniformity of Cell Size", "Cell Shape", "Marginal Adhesion", "Single Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses", "Class"]
    #print(dataset)
    dataset = dataset.sort_values(by=['Class'])
    class_dataset, folds = stratify_with_df(dataset)
    #class_counts = getClassCounts(dataset)
    #print(class_counts)

    #attribute_frequency_table = calculate_frequency_table(dataset)
    #print(attribute_frequency_table)

    print("------------------------------------")
    #attribute_likelihood_table = calculate_likelihood_table(attribute_frequency_table, class_counts)
    #print(attribute_likelihood_table)

    #class_probablity = calculate_class_probability(class_dataset)
    #print(class_probablity)
    
    #classifiers = naive_bayes_classify(attribute_likelihood_table, class_probablity)
    cross_validation(class_dataset, folds)
    
  
   
    
main()
