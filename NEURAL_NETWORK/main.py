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


class NeuralNetwork:
    def __init__(self):
        self.number = 7

    def main(self):
        # import data into dataframes
        print("IMPORTING DATA...")
        cancer_labels = ["Sample code number", "Uniformity of Clump Thickness", "Uniformity of Cell Size", "Cell Shape", "Marginal Adhesion", "Single Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses", "class"]
        cancer_df = self.import_data("breast-cancer-wisconsin-cleaned.txt", cancer_labels)
        cancer_df.drop(columns=cancer_df.columns[0], axis=1, inplace=True)
        
        glass_labels = ['id-num','retractive-index','sodium','magnesium','aluminum','silicon','potasium','calcium','barium','iron','class']
        glass_df = self.import_data("glass.data", glass_labels)
        glass_df.drop(columns=glass_df.columns[0], axis=1, inplace=True)
        
        soy_labels = ["date","plant-stand","precip","temp","hail","crop-hist","area-damaged","severity","seed-tmt","germination","leaves","lodging","stem-cankers","canker-lesion","fruiting-bodies","external-decay","mycelium","int-discolor","sclerotia","fruit-pods","roots","class"]
        soy_df = self.import_data("soybean-small-cleaned.csv", soy_labels)

        abalone_labels = ["sex","length","diameter","height","whole_weight","shucked_weight","viscera_weight","shell_weight","rings"]
        abalone_df = self.import_data("abalone.data",abalone_labels)

        machine_labels = ["vendor_name","model","myct","mmin","mmax","cach","chmin","chmax","prp","erp"]
        machine_df = self.import_data("machine.data",machine_labels)
        machine_df = machine_df.drop(['vendor_name','model'], axis = 1)

        forestfires_df = pd.read_csv("forestfires.csv", sep=",")
        # forestfires_df['month'] = forestfires_df.apply(lambda x: x['1'] if x['month']=='jan')
        forestfires_df = self.cyclical_ordinals(forestfires_df)

        glass_df = self.bin_set(glass_df, [1,2,3,4,5,6,7,8,9,10,11,12])
        
        abalone_df = self.one_hot_code(abalone_df)

        # normalize regression data -1 to +1
        print("OLD DATASET")
        print(glass_df)
        print("\nNEW DATASET")
        new_glass_df = self.min_max_normalization(glass_df)
        print("\n--------------------------------\n")

        # print("OLD DATASET")
        # print(abalone_df)
        # print("\nNEW DATASET")
        # print(self.min_max_normalization(abalone_df))
        # print("\n--------------------------------\n")

        # print("OLD DATASET")
        # print(machine_df)
        # print("\nNEW DATASET")
        # print(self.min_max_normalization(machine_df))
        # print("\n--------------------------------\n")

        # get classification db classes
        cancer_classes = cancer_df['class'].unique()
        soy_classes = soy_df['class'].unique()
        glass_classes = [1,2,3,4,5,6,7]
        # print(glass_classes)

        glass_features = glass_labels[1:-1]
        # print(glass_features)

        print("STRATIFYING DATA AND CREATING TUNING & FOLDS...")
        # Create training and testing dataframes for classification data, as well as the tuning dataframe
        # cancer_training1,cancer_testing1,cancer_training2,cancer_testing2,cancer_training3,cancer_testing3,cancer_training4,cancer_testing4,cancer_training5,cancer_testing5,cancer_training6,cancer_testing6,cancer_training7,cancer_testing7,cancer_training8,cancer_testing8,cancer_training9,cancer_testing9,cancer_training10,cancer_testing10,cancer_tuning = self.stratify_and_fold_classification(cancer_df)
        glass_training1,glass_testing1,glass_training2,glass_testing2,glass_training3,glass_testing3,glass_training4,glass_testing4,glass_training5,glass_testing5,glass_training6,glass_testing6,glass_training7,glass_testing7,glass_training8,glass_testing8,glass_training9,glass_testing9,glass_training10,glass_testing10,glass_tuning = self.stratify_and_fold_classification(new_glass_df)
        soy_training1,soy_testing1,soy_training2,soy_testing2,soy_training3,soy_testing3,soy_training4,soy_testing4,soy_training5,soy_testing5,soy_training6,soy_testing6,soy_training7,soy_testing7,soy_training8,soy_testing8,soy_training9,soy_testing9,soy_training10,soy_testing10,soy_tuning = self.stratify_and_fold_classification(soy_df)

        # Create training and testing dataframes for regression data
        # abalone_training1,abalone_testing1,abalone_training2,abalone_testing2,abalone_training3,abalone_testing3,abalone_training4,abalone_testing4,abalone_training5,abalone_testing5,abalone_training6,abalone_testing6,abalone_training7,abalone_testing7,abalone_training8,abalone_testing8,abalone_training9,abalone_testing9,abalone_training10,abalone_testing10,abalone_tuning = self.stratify_and_fold_regression(abalone_df)
        # machine_training1,machine_testing1,machine_training2,machine_testing2,machine_training3,machine_testing3,machine_training4,machine_testing4,machine_training5,machine_testing5,machine_training6,machine_testing6,machine_training7,machine_testing7,machine_training8,machine_testing8,machine_training9,machine_testing9,machine_training10,machine_testing10,machine_tuning = self.stratify_and_fold_regression(machine_df)
        # forestfires_training1,forestfires_testing1,forestfires_training2,forestfires_testing2,forestfires_training3,forestfires_testing3,forestfires_training4,forestfires_testing4,forestfires_training5,forestfires_testing5,forestfires_training6,forestfires_testing6,forestfires_training7,forestfires_testing7,forestfires_training8,forestfires_testing8,forestfires_training9,forestfires_testing9,forestfires_training10,forestfires_testing10,forestfires_tuning = self.stratify_and_fold_regression(forestfires_df)

        self.multi_layer_feedforward_network(len(glass_training1)-1, 1, 2, 7, "classification", glass_training1, glass_testing1, glass_classes, 2, glass_features)
        # self.multi_layer_feedforward_network(len(glass_testing1)-1, 1, 4, 7, "classification", glass_testing1, glass_classes, 2, glass_features)

        # self.multi_layer_feedforward_network(len(glass_training2)-1, 1, 4, 7, "classification", glass_training2, glass_classes, 3, glass_features)
        # self.multi_layer_feedforward_network(len(glass_testing2)-1, 1, 4, 7, "classification", glass_training3, glass_classes, 4, glass_features)
        # self.multi_layer_feedforward_network(len(soy_training1)-1, 1, 4, 4, "classification", soy_training1, soy_labels, 2)


    # generic function to import data to pd and apply labels
    def import_data(self, data: str, labels: list) -> pd.DataFrame:
        # import data into dataframe
        df = pd.read_csv(data, sep=",", header=None)
        # label dataframe
        df.columns = labels
        return df

    # function to bin sets
    def bin_set(self, dataset: pd.DataFrame, labels: list):
        for col_name, col_data in dataset.iteritems():
                if col_name != "Sample code number" and col_name != "class" and col_name != "id":
                    # pd.cut(dataset[col_name], bins=bins, labels=labels, include_lowest=True)
                    dataset[col_name] = pd.cut(dataset[col_name], len(labels), labels=labels)
        return dataset

    # function to one-hot code abalone
    def one_hot_code(self, dataset: pd.DataFrame):
        one_hot = pd.get_dummies(dataset['sex'])
        dataset = dataset.drop('sex', axis = 1)
        dataset = dataset.join(one_hot)
        dataset = dataset.reindex(columns=["F","I","M","diameter","height","whole_weight","shucked_weight","viscera_weight","shell_weight","rings"])
        # print(dataset)
        return dataset

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

        training1,testing1,training2,testing2,training3,testing3,training4,testing4,training5,testing5,training6,testing6,training7,testing7,training8,testing8,training9,testing9,training10,testing10 = self.form_training_test_sets(strat1_df,strat2_df,strat3_df,strat4_df,strat5_df,strat6_df,strat7_df,strat8_df,strat9_df,strat10_df)
        return training1,testing1,training2,testing2,training3,testing3,training4,testing4,training5,testing5,training6,testing6,training7,testing7,training8,testing8,training9,testing9,training10,testing10,tuning_df

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
        training1,testing1,training2,testing2,training3,testing3,training4,testing4,training5,testing5,training6,testing6,training7,testing7,training8,testing8,training9,testing9,training10,testing10 = self.form_training_test_sets(fold1,fold2,fold3,fold4,fold5,fold6,fold7,fold8,fold9,fold10)
        return training1,testing1,training2,testing2,training3,testing3,training4,testing4,training5,testing5,training6,testing6,training7,testing7,training8,testing8,training9,testing9,training10,testing10,tuning_df

    # function to combine folds into training and testing sets
    def form_training_test_sets(self, fold1: pd.DataFrame, fold2: pd.DataFrame,fold3: pd.DataFrame, fold4: pd.DataFrame, fold5: pd.DataFrame, fold6: pd.DataFrame, fold7: pd.DataFrame, fold8: pd.DataFrame, fold9: pd.DataFrame, fold10: pd.DataFrame) -> Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame]:
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

    # normalize numerical attributes to be in the range of -1 to +1
    def min_max_normalization(self, dataset: pd.DataFrame,):
        df = dataset.copy()
        for col_name, col_data in df.iteritems():
            if col_name != 'class' and col_name != 'rings' and col_name != 'erp' and col_name != 'area':
                x_max = df[col_name].loc[df[col_name].idxmax()]
                # x_max = df[col_name].agg(['min', 'max'])
                x_min = df[col_name].loc[df[col_name].idxmin()]
                df[col_name] = df[col_name].apply(lambda x: 2*((x - x_min)/(x_max - x_min))-1)

        return df

    # the weighted sum of the inputs for hidden nodes z_h 
    def sum_weight_for_hidden_nodes(self, weight, inputs, h):
        hidden = 0
        j = 0
        print(pd.DataFrame(inputs).T)
        for index, input in inputs.iteritems():
            if index != "class":
                #print("j:" , j)
                #print(weight[h-1, j])
                hidden += weight[h-1, j] * input
                print(hidden,"+=",weight[h-1, j],"*",input)
                j += 1
        print("hidden", hidden)

        return hidden

    # the activation function applied a the hidden node, sigmoid function 
    def sigmoid(self, a):
        z_h = 1 / (1 + np.exp(-a))
        print("zh", z_h)
        return z_h
    
    # if there is just one output unit, then we computes sum weight for output node
    def sum_weight_for_output_nodes(self, weight, hiddens, i):
        # print("weight", weight)
        # print("hiddend:", hiddens)
        # print(weight)
        # print(hiddens)
        output = 0
        for key, h_value in hiddens.items():
            print("key", key)
            print("value", h_value)
            output += weight[i, key] * h_value
        # print(output)
        #print("output", output)
        return output

    # the update rule for classification with 2 classes
    def check_k_classes_vih(self, r, y, mui, zh):
        # print("r:", r ,",", "y", y, ", mui:", mui, "," , "zh:", zh)
        delta_vh = mui * (r - y) * zh
        return delta_vh
    
    def check_k_classes_whj(self, mui, r, y, zh, vih, x, row, j_num, k, h):
        # print("r:", r ,",", "y", y, ", mui:", mui, "," , "zh:", zh, ", vh:", vh, ", x: ", row[x])
        # get sum
        sum = 0
        # print(vih)
        for i in range(1,k+1):
            # print(i,",",h)
            # print(vih[i,h])
            sum += (r-y[i]) * vih[i,h]
        delta_whj = mui * sum * zh * (1 - zh) * row[x]
        return delta_whj


    # perform linear activation function for regression node
    def linear_activation_function(self, node):
        pass

    # perform softmax activation function for classification node
    def softmax_activaiton(self, node):
        pass

    # sigmoid activation function for the hidden nodes
    # can choose to implement logostic or hyperbolic tnagent version
    # implement such that momentum is provided as an option
    def sigmoid_activation(self, node):
        pass

    # create multi-layer feedforward network with backpropogation 
    # Capable of training a network with arbitrary given number of inputs,
    # number of hidden layers, number of hidden units by layer, and number of outputs
    def multi_layer_feedforward_network(self, num_inputs: int, num_hidden_layers: int, num_hidden_units: int, num_outputs: int, version: str, train_df:pd.DataFrame, test_df:pd.DataFrame, class_list:list, num_iterations:int, feature_labels: list):
        counter = 0
        while(counter < num_iterations):
            vih = np.random.uniform(-0.01, 0.01, (len(class_list)+1, num_hidden_units))
            # print(vih)
            whj = np.random.uniform(-0.01, 0.01, (num_hidden_units, len(feature_labels)))
            # print(whj)
            
            mui = 0.1
            train_df = train_df.copy()
            shuffed_inputs = train_df.sample(frac=1)
            hidden_dict = {}
            output_dict = {}
            yi_dict = {}
            hidden_dict[0] = 1
            delta_vih_dict = {}
            delta_whj_dict = {}
            # print("before", yi_dict)
            stop = 0
            for row_label, row in shuffed_inputs.iterrows():
                hidden_dict = {}
                output_dict = {}
                yi_dict = {}
                hidden_dict[0] = 1
                delta_vih_dict = {}
                delta_whj_dict = {}
                # print("before vih", vih)
                # print("before whj,",whj)
                self.processing_at_node(num_hidden_units, class_list,  vih, whj, hidden_dict, row, output_dict, yi_dict)
                print("after vih", vih)
                print("after whj", whj)
                print("after oi:", output_dict)
                print("after yi:", yi_dict)
                print("before zh:", hidden_dict, "\n")
                # start: calculate weight updates for v_ih
                for i in range(1, len(class_list)+1):
                    for h in range(0,num_hidden_units):
                        delta_vh = self.check_k_classes_vih(row[-1], yi_dict[i], mui, hidden_dict[h])
                        delta_vih_dict[(i,h)] = delta_vh
                #print("delta_vh_dict", delta_vh_dict)
                # end: updating weights for v_ih

                # begin: calculate updates for w_hj
                # print("-------------------------------------")
                # print(hidden_dict)
                # print(vih)
                # print(yi_dict)
                for h in range(1, num_hidden_units):
                    j_num = 0
                    for j in feature_labels:
                        # print(j,":",h,",",j_num)
                        # print(row)
                        # print(row[-1])
                        # print(yi_dict)
                        # print(yi_dict[row[-1]])
                        # print(hidden_dict[h])
                        delta_whj = self.check_k_classes_whj(mui, row[-1], yi_dict, hidden_dict[h], vih, j, row, j_num, len(class_list), h)
                        delta_whj_dict[(h,j_num)] = delta_whj
                        j_num += 1
                # end: calculate updates for w_hj

                # begin: apply weight updates for v_ih
                for i in range(1, len(class_list)+1):
                    for h in range(num_hidden_units):
                        vih = vih + delta_vih_dict[(i,h)]
                # end: apply weight updates for v_ih

                # print(delta_vih_dict)
                # print("---------------------------------")
                # print(delta_whj_dict)

                # begin: apply updates for w_hj
                for h in range(1, num_hidden_units):
                    for j in range(len(feature_labels)):
                        whj = whj + delta_whj_dict[(h,j)]
                # end: apply updates for w_hj
                # stop += 1
                # if stop == 1000:
                #     sys.exit(0)

            # end of epoch, update counter
            counter += 1
            print(counter)
        print("exited while")
        # predict test classes
        print("RUNNING TEST")
        test_df = test_df.copy()
        test_shuffed_inputs = test_df.sample(frac=1)
        for row_label, row in test_shuffed_inputs.iterrows():
            predict_yi = self.processing_at_node(num_hidden_units, class_list,  vih, whj, hidden_dict, row, output_dict, yi_dict)
            # print(row)
            # print(yi_dict, "\n")
            max_yi = max(predict_yi, key=predict_yi.get)
            test_shuffed_inputs.at[row_label,'Classifier'] = max_yi
        print("-------------------------------------------------------\n")
        print(test_shuffed_inputs)
        return vih, whj

    # for the nodes in network, process weight inputs to produce output
    def processing_at_node(self,
                           num_hidden_units: int, 
                           class_list:list, 
                           vih: np.matrix,
                           whj: np.matrix,
                           hidden_dict: dict,
                           row: pd.Series,
                           output_dict: dict,
                           yi_dict: dict):
        # start: hidden output weights
        zh = 0
        for h in range(1, num_hidden_units):
            hidden_weight = self.sum_weight_for_hidden_nodes(whj, row, h)
            zh = self.sigmoid(hidden_weight)
            #print("zh", zh)
            hidden_dict[h] = zh
        # end: hidden output weights

        total = 0
        # start: output weights
        for i in range(1, len(class_list)+1):
            #print("Class",i)
            # oi = 0
            # oi += self.sum_weight_for_output_nodes(vih, hidden_dict, i-1)
            oi = self.sum_weight_for_output_nodes(vih, hidden_dict, i-1)
            output_dict[i] = oi
            print("OI",oi)
            print("exp(oi)", np.exp(oi))
            print("Old total", total)
            total = total + np.exp(oi)
            print("new total", total)
            # oi = 0
        #print("total: ", total)
        # end: output weights
        
        # print("Oi:",output_dict)
        print("Total:",total)
        # start: actual output
        for i in range(1, len(class_list)+1):
            yi = np.exp(output_dict[i]) / total
            yi_dict[i] = yi
        return yi_dict

nn = NeuralNetwork()
nn.main()
print("Hooray")
