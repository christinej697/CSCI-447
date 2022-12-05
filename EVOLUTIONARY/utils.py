# Class to implement to KNN
from calendar import month
import math
from random import random, uniform
import pandas as pd
import numpy as np
from typing import Tuple
from statistics import mode
from termcolor import colored
import sys
from mlp import MLP

# function to combine folds into training and testing sets
def form_training_test_sets(fold1: pd.DataFrame, fold2: pd.DataFrame,fold3: pd.DataFrame, fold4: pd.DataFrame, fold5: pd.DataFrame, fold6: pd.DataFrame, fold7: pd.DataFrame, fold8: pd.DataFrame, fold9: pd.DataFrame, fold10: pd.DataFrame) -> Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    training1 = pd.concat([fold1,fold2,fold3,fold4,fold5,fold6,fold7,fold8,fold9])
    testing1 = fold10.copy()
    training2 = pd.concat([fold10,fold1,fold2,fold3,fold4,fold5,fold6,fold7,fold8])
    testing2 = fold9.copy()
    training3 = pd.concat([fold9,fold10,fold1,fold2,fold3,fold4,fold5,fold6,fold7])
    testing3 = fold8.copy()
    training4 = pd.concat([fold8,fold9,fold10,fold1,fold2,fold3,fold4,fold5,fold6])
    testing4 = fold7.copy()
    training5 = pd.concat([fold7,fold8,fold9,fold10,fold1,fold2,fold3,fold4,fold5])
    testing5 = fold6.copy()
    training6 = pd.concat([fold6,fold7,fold8,fold9,fold10,fold1,fold2,fold3,fold4])
    testing6 = fold5.copy()
    training7 = pd.concat([fold5,fold6,fold7,fold8,fold9,fold10,fold1,fold2,fold3])
    testing7 = fold4.copy()
    training8 = pd.concat([fold4,fold5,fold6,fold7,fold8,fold9,fold10,fold1,fold2])
    testing8 = fold3.copy()
    training9 = pd.concat([fold3,fold4,fold5,fold6,fold7,fold8,fold9,fold10,fold1])
    testing9 = fold2.copy()
    training10 = pd.concat([fold2,fold3,fold4,fold5,fold6,fold7,fold8,fold9,fold10])
    testing10 = fold1.copy()
    return training1,testing1,training2,testing2,training3,testing3,training4,testing4,training5,testing5,training6,testing6,training7,testing7,training8,testing8,training9,testing9,training10,testing10

def get_predicted_class(confusion_matrix, class_names, actual_class, predicted_class):
    for name in class_names:
        confusion_matrix[name] = {"TP":0, "FP":0, "FN": 0, "TN":0}
        index = 0
        for act in actual_class:
            if str(act) == str(name) and str(predicted_class[index]) == str(name):
                predicated = "TP"
            if str(act) == str(name) and str(predicted_class[index]) != str(name):
                predicated = "FN"
            if  str(act) != str(name) and str(predicted_class[index]) == str(name):
                predicated = "FP"
            if str(act) != str(name) and str(predicted_class[index]) != str(name):
                predicated = "TN"
            confusion_matrix[name][predicated] += 1
            index += 1
    return confusion_matrix

class UTILS:
    def __init__(self):
        self.number = 7
    
    # generic function to import data to pd and apply labels
    def import_data(self, data: str, labels: list) -> pd.DataFrame:
        # import data into dataframe
        df = pd.read_csv(data, sep=",", header=None)
        # label dataframe
        df.columns = labels
        return df

    # function to implement stratified tuning and 10 fold for regression data 
    def stratify_and_fold_regression(self, dataset: pd.DataFrame) -> Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame, pd.DataFrame]:
        sorted_df = dataset.sort_values(dataset.columns[-1])
        group_list = np.array_split(sorted_df,10)
        training,tuning = ([] for i in range(2))
        # split into 10% for tuning and 90% for 10-folds
        flag=True
        for g in group_list:
            for i in range((int(g.shape[0]/10)) + 1):
                if (1+i*10) <= (g.shape[0]):
                    training.append(g.iloc[i*10:9+i*10])
                if (9+i*10) <= (g.shape[0]):
                    tuning.append(g.iloc[[9+i*10]])
            flag = False
        training_df = pd.concat(training)
        tuning_df = pd.concat(tuning)
        fold_list = np.array_split(training_df,10)
        strat1,strat2,strat3,strat4,strat5,strat6,strat7,strat8,strat9,strat10 = ([] for i in range(10))
        for fold in fold_list:
            for i in range(fold.shape[0]):
                if (i*10) <= (fold.shape[0] - 1):
                    strat1.append(fold.iloc[[i*10]])
                if (1+i*10) <= (fold.shape[0] - 1):
                    strat2.append(fold.iloc[[1+i*10]])
                if (2+i*10) <= (fold.shape[0] - 1):
                    strat3.append(fold.iloc[[2+i*10]])
                if (3+i*10) <= (fold.shape[0] - 1):
                    strat4.append(fold.iloc[[3+i*10]])
                if (4+i*10) <= (fold.shape[0] - 1):
                    strat5.append(fold.iloc[[4+i*10]])
                if (5+i*10) <= (fold.shape[0] - 1):
                    strat6.append(fold.iloc[[5+i*10]])
                if (6+i*10) <= (fold.shape[0] - 1):
                    strat7.append(fold.iloc[[6+i*10]])
                if (7+i*10) <= (fold.shape[0] - 1):
                    strat8.append(fold.iloc[[7+i*10]])
                if (8+i*10) <= (fold.shape[0] - 1):
                    strat9.append(fold.iloc[[8+i*10]])
                if (9+i*10) <= (fold.shape[0] - 1):
                    strat10.append(fold.iloc[[9+i*10]])
        strat1_df = pd.concat(strat1)
        strat2_df = pd.concat(strat2)
        strat3_df = pd.concat(strat3)
        strat4_df = pd.concat(strat4)
        strat5_df = pd.concat(strat5)
        strat6_df = pd.concat(strat6)
        strat7_df = pd.concat(strat7)
        strat8_df = pd.concat(strat8)
        strat9_df = pd.concat(strat9)
        strat10_df = pd.concat(strat10)

        training1,testing1,training2,testing2,training3,testing3,training4,testing4,training5,testing5,training6,testing6,training7,testing7,training8,testing8,training9,testing9,training10,testing10 = form_training_test_sets(strat1_df,strat2_df,strat3_df,strat4_df,strat5_df,strat6_df,strat7_df,strat8_df,strat9_df,strat10_df)
        return training1,testing1,training2,testing2,training3,testing3,training4,testing4,training5,testing5,training6,testing6,training7,testing7,training8,testing8,training9,testing9,training10,testing10,tuning_df

    # normalize numerical attributes to be in the range of -1 to +1
    def min_max_normalization(self, dataset: pd.DataFrame,):
        df = dataset.copy()
        for col_name, col_data in df.iteritems():
            if col_name != "class":
                x_max = df[col_name].loc[df[col_name].idxmax()]
                # x_max = df[col_name].agg(['min', 'max'])
                x_min = df[col_name].loc[df[col_name].idxmin()]
                df[col_name] = df[col_name].apply(lambda x: 2*((x - x_min)/(x_max - x_min))-1)
        return df

    # function to one-hot code abalone
    def one_hot_code(self, dataset: pd.DataFrame, col_name):
        if col_name == 'sex':
            one_hot = pd.get_dummies(dataset['sex'],prefix="class_")
            dataset = dataset.drop('sex', axis = 1)
            dataset = dataset.join(one_hot)
            dataset = dataset.reindex(columns=["F","I","M","diameter","height","whole_weight","shucked_weight","viscera_weight","shell_weight","rings"])
        elif col_name == 'class':
            one_hot = pd.get_dummies(dataset['class'])
            dataset = dataset.drop('class', axis = 1)
            dataset = dataset.join(one_hot)
        return dataset

    # function to implement stratified tuning and 10 fold for classification data
    def stratify_and_fold_classification(self, dataset: pd.DataFrame) -> Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame, pd.DataFrame]:
        classes = dataset['class'].unique()
        # randomizing the data set
        dataset = dataset.reindex(np.random.permutation(dataset.index)) 
        #reset the index
        dataset = dataset.reset_index(drop=True)
        # split classes
        class_df_set={}
        for c in classes:
            class_df_set[str(c)+'_df'] = dataset[dataset['class'] == c]
        # split into 10% for tuning and 90% for 10-folds
        class_sets = {}
        tuning_sets ={}
        for key,value in class_df_set.items():
            fold1,fold2,fold3,fold4,fold5,fold6,fold7,fold8,fold9,fold10 = np.array_split(value,10)
            class_sets[key+'_folds']=pd.concat([fold2,fold3,fold4,fold5,fold6,fold7,fold8,fold9,fold10])
            tuning_sets[key] = fold1.copy()
        tuning_df = pd.concat(tuning_sets.values())
        # make 10 different folds of the same size
        class_fold_set = {}
        for key,value in class_sets.items():
            fold1,fold2,fold3,fold4,fold5,fold6,fold7,fold8,fold9,fold10 = np.array_split(class_sets[key],10)
            class_fold_set[key]=[fold1,fold2,fold3,fold4,fold5,fold6,fold7,fold8,fold9,fold10]
            flag = False
        for key,value in class_fold_set.items():
            if flag == False:
                fold1,fold2,fold3,fold4,fold5,fold6,fold7,fold8,fold9,fold10 = value[0],value[1],value[2],value[3],value[4],value[5],value[6],value[7],value[8],value[9]
                flag = True
            else:
                fold1 = pd.concat([fold1,value[0]])
                fold2 = pd.concat([fold2,value[1]])
                fold3 = pd.concat([fold3,value[2]])
                fold4 = pd.concat([fold4,value[3]])
                fold5 = pd.concat([fold5,value[4]])
                fold6 = pd.concat([fold6,value[5]])
                fold7 = pd.concat([fold7,value[6]])
                fold8 = pd.concat([fold8,value[7]])
                fold9 = pd.concat([fold9,value[8]])
                fold10 = pd.concat([fold10,value[9]])
        training1,testing1,training2,testing2,training3,testing3,training4,testing4,training5,testing5,training6,testing6,training7,testing7,training8,testing8,training9,testing9,training10,testing10 = form_training_test_sets(fold1,fold2,fold3,fold4,fold5,fold6,fold7,fold8,fold9,fold10)
        return training1,testing1,training2,testing2,training3,testing3,training4,testing4,training5,testing5,training6,testing6,training7,testing7,training8,testing8,training9,testing9,training10,testing10,tuning_df

    # loop through and change it to integers
    def one_hot_code(self, classes):
        class_dict = {}
        class_dict["D1"] = 1
        class_dict["D2"] = 2
        class_dict["D3"] = 3
        class_dict["D4"] = 4
        for i in range(len(classes.values)):
            for key, value in class_dict.items():
                if classes[i] == key:
                    classes[i] = value

        # function to one-hot code abalone
    def one_hot_code_abalone(self, dataset: pd.DataFrame, col_name):
        if col_name == 'sex':
            one_hot = pd.get_dummies(dataset['sex'])
            dataset = dataset.drop('sex', axis = 1)
            dataset = dataset.join(one_hot)
            dataset = dataset.reindex(columns=["F","I","M","diameter","height","whole_weight","shucked_weight","viscera_weight","shell_weight","rings"])
        elif col_name == 'class':
            one_hot = pd.get_dummies(dataset['class'])
            dataset = dataset.drop('class', axis = 1)
            dataset = dataset.join(one_hot)
        return dataset

    #function to create confusion matrix
    def calculate_loss_function(self, classified_df, class_names, version):
        classified_df = classified_df.copy()
        confusion_matrix = {}

        actual_class = classified_df["class"].tolist()

        if version == "classification":
            predicted_class = classified_df["prediction"].tolist()
            confusion_matrix = get_predicted_class(confusion_matrix, class_names, actual_class, predicted_class)
        elif version == "regression":
            predicted_class = classified_df["prediction"].tolist()
            confusion_matrix = get_predicted_class(confusion_matrix, class_names, actual_class, predicted_class)
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_tn = 0
        total = 0
        for name in class_names:
            total_tp += confusion_matrix[name]["TP"]
            total_fp += confusion_matrix[name]["FP"]
            total_fn += confusion_matrix[name]["FN"]
            total_tn += confusion_matrix[name]["TN"]
        total += total_tn + total_tp + total_fp + total_fn
        precision = total_tp / (total_tp + total_fp)
        recall = total_tp / (total_tp + total_fn)
        if precision == 0 and recall == 0:
            F1 = 0
        else:
            F1 = 2 * ((precision * recall) / (precision + recall))
        accuracy = (total_tp + total_tn) / total
        loss = {"Accuracy/0-1": accuracy, "Precision": precision, "Recall": recall, "F1": F1}
        return loss

    def calculate_loss_for_regression(self, classified_df):
        sigma = 0.5
        classified_df = classified_df.copy()
        predicted_class = classified_df["Prediction"].tolist()
        actual_class = classified_df.iloc[:,-2].tolist()
        # print(predicted_class)
        # print(actual_class)
        loss = 0
        sum = 0
        all_points = []
        sigma = []
        for i in range(len(actual_class)):
            all_points.append(math.fabs(actual_class[i] - predicted_class[i]))
            sum += math.fabs(actual_class[i] - predicted_class[i])
        loss = sum / len(actual_class)
        loss_list = {}
        T_couter = 0
        F_couter = 0
        for i in all_points:
            if i <= loss:
                T_couter += 1
            else:
                F_couter += 1
        loss_list["Correct Prediction"] = (T_couter/len(predicted_class)) * 100 
        loss_list["Incorrect Prediction"] =  (F_couter/len(predicted_class)) * 100
        # print(loss_list)
        return loss

    def get_loss(self, performances, classes):
        loss_dict = {}
        loss_sum = 0
        couter = 1
        best_f1 = 0
        best_num = 1
        for i in performances:
            loss = self.calculate_loss_np(self, i, classes)
            if loss["F1"] > best_f1:
                best_f1 = loss["F1"]
                best_num = couter
            loss_sum += loss['F1']
            loss_dict[couter] = loss
            #print("test case number: {}, loss: {}".format(couter, loss))
            couter += 1
        #print(loss_sum)
        avg_p = loss_sum / couter
        # print()
        # print("The average F1 score of 10 folds is: ", avg_p)
        return loss_dict, best_num

    def calculate_loss_np(self, output, classes, version = "class"):
        loss = {}
        confusion_matrix = {}
        get_predicted_class(confusion_matrix, classes, classes, output)
        # print(confusion_matrix)
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_tn = 0
        total = 0
        for name in classes:
            total_tp += confusion_matrix[name]["TP"]
            total_fp += confusion_matrix[name]["FP"]
            total_fn += confusion_matrix[name]["FN"]
            total_tn += confusion_matrix[name]["TN"]
        total += total_tn + total_tp + total_fp + total_fn
        if total_tp == 0 and total_fp == 0:
            precision = 0
        else:
            precision = total_tp / (total_tp + total_fp)
        if total_tp == 0 and total_fn == 0:
            recall = 0
        else:
            recall = total_tp / (total_tp + total_fn)
        if precision == 0 and recall == 0:
            F1 = 0
        else:
            F1 = 2 * ((precision * recall) / (precision + recall))
        accuracy = (total_tp + total_tn) / total
        loss = {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1": F1}
        return loss

    def find_max_value(self, output, classes):
        idx = []
        for row in output:
            max = np.max(row)
            index= row.tolist().index(max)
            idx.append(classes[index])
        return idx

    def get_performance(self, mlp: MLP, member, classes: list, input):
        mlp.set_weights(member)
        output = mlp.forward_feed(input)
        result = self.find_max_value(self, output, classes)
        return result

    # function for forest fires cyclical ordinals
    def cyclical_ordinals(self, df: pd.DataFrame):
        # print("Entering Cyc")
        new_df = df.copy()
        # replace months with integers
        new_df.loc[new_df['month'] == 'jan', 'month'] = 1
        new_df.loc[new_df['month'] == 'feb', 'month'] = 2
        new_df.loc[new_df['month'] == 'mar', 'month'] = 3
        new_df.loc[new_df['month'] == 'apr', 'month'] = 4
        new_df.loc[new_df['month'] == 'may', 'month'] = 5
        new_df.loc[new_df['month'] == 'jun', 'month'] = 6
        new_df.loc[new_df['month'] == 'jul', 'month'] = 7
        new_df.loc[new_df['month'] == 'aug', 'month'] = 8
        new_df.loc[new_df['month'] == 'sep', 'month'] = 9
        new_df.loc[new_df['month'] == 'oct', 'month'] = 10
        new_df.loc[new_df['month'] == 'nov', 'month'] = 11
        new_df.loc[new_df['month'] == 'dec', 'month'] = 12
        # replace days with integers
        new_df.loc[new_df['day'] == 'sun', 'day'] = 1
        new_df.loc[new_df['day'] == 'mon', 'day'] = 2
        new_df.loc[new_df['day'] == 'tue', 'day'] = 3
        new_df.loc[new_df['day'] == 'wed', 'day'] = 4
        new_df.loc[new_df['day'] == 'thu', 'day'] = 5
        new_df.loc[new_df['day'] == 'fri', 'day'] = 6
        new_df.loc[new_df['day'] == 'sat', 'day'] = 7
        # change values to cyclic using cosine
        month_norm = 2 * math.pi * new_df["month"] / new_df["month"].max()
        month_norm = month_norm.to_numpy().astype('float64')
        new_df["cos_month"] = np.cos(month_norm)
        day_norm = 2 * math.pi * new_df["day"] / new_df["day"].max()
        day_norm = day_norm.to_numpy().astype('float64')
        new_df["cos_day"] = np.cos(day_norm)
        new_df = new_df.drop('month', axis = 1)
        new_df = new_df.drop('day', axis = 1)
        new_df = new_df.reindex(columns=["X","Y",'cos_month','cos_day','FFMC','DMC','DC','ISI','temp','RH','wind','rain','area'])
        return new_df
