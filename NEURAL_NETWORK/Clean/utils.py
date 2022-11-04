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

            # function to one-hot code abalone
    def one_hot_code(self, dataset: pd.DataFrame, col_name):
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

