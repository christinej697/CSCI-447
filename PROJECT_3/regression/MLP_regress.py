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
import statistics


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
        # print(glass_features)
        forestfires_df = pd.read_csv("forestfires.csv", sep=",")
        # forestfires_df['month'] = forestfires_df.apply(lambda x: x['1'] if x['month']=='jan')
        forestfires_df = self.cyclical_ordinals(forestfires_df)

        glass_df = self.bin_set(glass_df, [1,2,3,4,5,6,7,8,9,10,11,12])
        
        abalone_df = self.one_hot_code(abalone_df, 'sex')
        # print(abalone_df)

        # get classification db classes
        cancer_classes = cancer_df['class'].unique()
        soy_classes = soy_df['class'].unique()
        glass_classes = [1,2,3,5,6,7]
        # print(glass_classes)

        glass_features = glass_labels[1:-1]
        cancer_features = cancer_labels[0:-1]
        abalone_features = abalone_labels[0:-1]
        machine_features = machine_labels[0:-1]
        soy_features = soy_labels[0:-1]
        forestfires_features = list(forestfires_df)[0:-1]
        

        # normalize classification data -1 to +1
        soy_df = self.min_max_normalization(soy_df)
        # new_soy_df = self.one_hot_code(soy_df,'class')
        new_glass_df = self.min_max_normalization(glass_df)
        # new_glass_df = self.one_hot_code(new_glass_df,'class')
        # print(self.one_hot_code(new_glass_df, 'class'))
        # cancer_df = self.min_max_normalization(cancer_df)
        # # cancer_df = self.one_hot_code(cancer_df,'class')
        # # cancer_df = self.class_normalization(cancer_df)

        # # normalize regression data -1 to +1]
        forestfires_df = self.min_max_normalization(forestfires_df)
        abalone_df = self.min_max_normalization(abalone_df)
        machine_df = self.min_max_normalization(machine_df)

        print("STRATIFYING DATA AND CREATING TUNING & FOLDS...")
        # Create training and testing dataframes for classification data, as well as the tuning dataframe
        # cancer_training1,cancer_testing1,cancer_training2,cancer_testing2,cancer_training3,cancer_testing3,cancer_training4,cancer_testing4,cancer_training5,cancer_testing5,cancer_training6,cancer_testing6,cancer_training7,cancer_testing7,cancer_training8,cancer_testing8,cancer_training9,cancer_testing9,cancer_training10,cancer_testing10,cancer_tuning = self.stratify_and_fold_classification(cancer_df)
        glass_training1,glass_testing1,glass_training2,glass_testing2,glass_training3,glass_testing3,glass_training4,glass_testing4,glass_training5,glass_testing5,glass_training6,glass_testing6,glass_training7,glass_testing7,glass_training8,glass_testing8,glass_training9,glass_testing9,glass_training10,glass_testing10,glass_tuning = self.stratify_and_fold_classification(new_glass_df)
        soy_training1,soy_testing1,soy_training2,soy_testing2,soy_training3,soy_testing3,soy_training4,soy_testing4,soy_training5,soy_testing5,soy_training6,soy_testing6,soy_training7,soy_testing7,soy_training8,soy_testing8,soy_training9,soy_testing9,soy_training10,soy_testing10,soy_tuning = self.stratify_and_fold_classification(soy_df)
        # for dataset in [soy_training1,soy_testing1,soy_training2,soy_testing2,soy_training3,soy_testing3,soy_training4,soy_testing4,soy_training5,soy_testing5,soy_training6,soy_testing6,soy_training7,soy_testing7,soy_training8,soy_testing8,soy_training9,soy_testing9,soy_training10,soy_testing10,soy_tuning]:
        #     print(dataset)
        #     dataset = self.one_hot_code(dataset, 'class')
        #     print(dataset)
        # print(soy_tuning)
        # sys.exit(0)
        # soy_list = [soy_training1,soy_testing1,soy_training2,soy_testing2,soy_training3,soy_testing3,soy_training4,soy_testing4,soy_training5,soy_testing5,soy_training6,soy_testing6,soy_training7,soy_testing7,soy_training8,soy_testing8,soy_training9,soy_testing9,soy_training10,soy_testing10,soy_tuning]
        # soy_training1,soy_testing1,soy_training2,soy_testing2,soy_training3,soy_testing3,soy_training4,soy_testing4,soy_training5,soy_testing5,soy_training6,soy_testing6,soy_training7,soy_testing7,soy_training8,soy_testing8,soy_training9,soy_testing9,soy_training10,soy_testing10,soy_tuning = [self.one_hot_code(df, 'class') for df in soy_list]
        # print(soy_training1)
        # print(soy_testing2)
        # sys.exit(0)

        # Create training and testing dataframes for regression data
        abalone_training1,abalone_testing1,abalone_training2,abalone_testing2,abalone_training3,abalone_testing3,abalone_training4,abalone_testing4,abalone_training5,abalone_testing5,abalone_training6,abalone_testing6,abalone_training7,abalone_testing7,abalone_training8,abalone_testing8,abalone_training9,abalone_testing9,abalone_training10,abalone_testing10,abalone_tuning = self.stratify_and_fold_regression(abalone_df)
        machine_training1,machine_testing1,machine_training2,machine_testing2,machine_training3,machine_testing3,machine_training4,machine_testing4,machine_training5,machine_testing5,machine_training6,machine_testing6,machine_training7,machine_testing7,machine_training8,machine_testing8,machine_training9,machine_testing9,machine_training10,machine_testing10,machine_tuning = self.stratify_and_fold_regression(machine_df)
        forestfires_training1,forestfires_testing1,forestfires_training2,forestfires_testing2,forestfires_training3,forestfires_testing3,forestfires_training4,forestfires_testing4,forestfires_training5,forestfires_testing5,forestfires_training6,forestfires_testing6,forestfires_training7,forestfires_testing7,forestfires_training8,forestfires_testing8,forestfires_training9,forestfires_testing9,forestfires_training10,forestfires_testing10,forestfires_tuning = self.stratify_and_fold_regression(forestfires_df)
        # print(abalone_training1)
        # self.multi_layer_feedforward_network(len(abalone_training1)-1, 1, 4, 1, "regression", abalone_training1, abalone_testing1, ['rings'], 2, abalone_features)
        # print(machine_testing1)

############################ Video Requirement 1A and 6 #######################################################

        # loss1 = self.run_machine_mlps(1,machine_training1, machine_testing1, machine_features)
        # loss2 = self.run_machine_mlps(2,machine_training2, machine_testing2, machine_features)
        # loss3 = self.run_machine_mlps(3,machine_training3, machine_testing3, machine_features)
        # print("Regression: Machine 1 Fold Output\n----------------------------------------")
        # loss4 = self.run_machine_mlps(4,machine_training4, machine_testing4, machine_features,True)
        # loss5 = self.run_machine_mlps(5,machine_training5, machine_testing5, machine_features)
        # loss6 = self.run_machine_mlps(6,machine_training6, machine_testing6, machine_features)
        # loss7 = self.run_machine_mlps(7,machine_training7, machine_testing7, machine_features)
        # loss8 = self.run_machine_mlps(8,machine_training8, machine_testing8, machine_features)
        # loss9 = self.run_machine_mlps(9,machine_training9, machine_testing9, machine_features)
        # loss10 = self.run_machine_mlps(10,machine_training10, machine_testing2, machine_features)


        # average_0_MAPE = (loss1[0]['MAPE'] + loss2[0]['MAPE'] + loss3[0]['MAPE'] + loss4[0]['MAPE'] + loss5[0]['MAPE'] + loss6[0]['MAPE'] + loss7[0]['MAPE'] + loss8[0]['MAPE'] + loss9[0]['MAPE'] + loss10[0]['MAPE'])/10
        # average_1_MAPE = (loss1[1]['MAPE'] + loss2[1]['MAPE'] + loss3[1]['MAPE'] + loss4[1]['MAPE'] + loss5[1]['MAPE'] + loss6[1]['MAPE'] + loss7[1]['MAPE'] + loss8[1]['MAPE'] + loss9[1]['MAPE'] + loss10[1]['MAPE'])/10
        # average_2_MAPE = (loss1[2]['MAPE'] + loss2[2]['MAPE'] + loss3[2]['MAPE'] + loss4[2]['MAPE'] + loss5[2]['MAPE'] + loss6[2]['MAPE'] + loss7[2]['MAPE'] + loss8[2]['MAPE'] + loss9[2]['MAPE'] + loss10[2]['MAPE'])/10
        
        # average_0_MAE = (loss1[0]['MAE'] + loss2[0]['MAE'] + loss3[0]['MAE'] + loss4[0]['MAE'] + loss5[0]['MAE'] + loss6[0]['MAE'] + loss7[0]['MAE'] + loss8[0]['MAE'] + loss9[0]['MAE'] + loss10[0]['MAE'])/10
        # average_1_MAE = (loss1[1]['MAE'] + loss2[1]['MAE'] + loss3[1]['MAE'] + loss4[1]['MAE'] + loss5[1]['MAE'] + loss6[1]['MAE'] + loss7[1]['MAE'] + loss8[1]['MAE'] + loss9[1]['MAE'] + loss10[1]['MAE'])/10
        # average_2_MAE = (loss1[2]['MAE'] + loss2[2]['MAE'] + loss3[2]['MAE'] + loss4[2]['MAE'] + loss5[2]['MAE'] + loss6[2]['MAE'] + loss7[2]['MAE'] + loss8[2]['MAE'] + loss9[2]['MAE'] + loss10[2]['MAE'])/10

        # print("------------------------------- Regression: Machine Average Performances ---------------------------------")
        # print("Average MAPE for 0 layers:", average_0_MAPE)
        # print("Average MAPE for 1 layers:", average_1_MAPE)
        # print("Average MAPE for 2 layers:", average_2_MAPE)

        # print("\nAverage MAE for 0 layers:", average_0_MAE)
        # print("Average MAE for 1 layers:", average_1_MAE)
        # print("Average MAE for 2 layers:", average_2_MAE)


############################ Video Requirement 2 #######################################################
        
        norm_train, xmax, xmin = self.regress_class_normalization(machine_training3, 'erp')
        norm_test, xmax, xmin = self.regress_class_normalization(machine_testing3, 'erp')

        # print("MACHINE 0 LAYERS MODEL")
        # predicted = self.multi_layer_feedforward_network(len(norm_train)-1, 0, 10, 1, "regression", norm_train, norm_test, ['erp'], 4, machine_features, 0.01, 0.5, "sample_model")

        # print("\n MACHINE 1 LAYER MODEL")
        # predicted = self.multi_layer_feedforward_network(len(norm_train)-1, 1, 10, 1, "regression", norm_train, norm_test, ['erp'], 70, machine_features, 0.001, 0.5, "sample_model")

        # print("\nMACHINE 2 LAYER MODEL")
        # predicted = self.multi_layer_feedforward_network(len(norm_train)-1, 2, 10, 1, "regression", norm_train, norm_test, ['erp'], 70, machine_features, 0.0001, 0.9, "sample_model")

# ############################ Video Requirement 3 #######################################################
        
        # print("\n MACHINE 2 LAYER PROPOGATION")
        # predicted = self.multi_layer_feedforward_network(len(norm_train)-1, 2, 10, 1, "regression", norm_train, norm_test, ['erp'], 70, machine_features, 0.0001, 0.9, "propogate")

# ############################ Video Requirement 4 #######################################################
        
        print("\n MACHINE 2 LAYER GRADIENT AND WEIGHT UPDATES")
        predicted = self.multi_layer_feedforward_network(len(norm_train)-1, 2, 10, 1, "regression", norm_train, norm_test, ['erp'], 70, machine_features, 0.0001, 0.9, "gradient")


##############################################################################################

        # self.multi_layer_feedforward_network(len(glass_training1)-1, 1, 4, 7, "multiple_classification", glass_training1, glass_testing1, glass_classes, 20, glass_features)
        # self.multi_layer_feedforward_network(len(cancer_training1)-1, 1, 10, 1, "binary_classification", cancer_training1, cancer_testing1, cancer_classes, 2, cancer_features)
        # self.multi_layer_feedforward_network(len(glass_testing1)-1, 1, 4, 7, "classification", glass_testing1, glass_classes, 2, glass_features)

        # self.multi_layer_feedforward_network(len(glass_training2)-1, 1, 4, 7, "classification", glass_training2, glass_classes, 3, glass_features)
        # self.multi_layer_feedforward_network(len(glass_testing2)-1, 1, 4, 7, "classification", glass_training3, glass_classes, 4, glass_features)
        # print(machine_testing1)
        # normed, xmax, xmin = self.regress_class_normalization(machine_testing1, 'erp')
        # print(normed)
        # print(self.regress_undo_normalization(normed, 'erp', xmax, xmin))

        # glass_list = [glass_training1,glass_testing1,glass_training2,glass_testing2,glass_training3,glass_testing3,glass_training4,glass_testing4,glass_training5,glass_testing5,glass_training6,glass_testing6,glass_training7,glass_testing7,glass_training8,glass_testing8,glass_training9,glass_testing9,glass_training10,glass_testing10,glass_tuning]
        # glass_training1,glass_testing1,glass_training2,glass_testing2,glass_training3,glass_testing3,glass_training4,glass_testing4,glass_training5,glass_testing5,glass_training6,glass_testing6,glass_training7,glass_testing7,glass_training8,glass_testing8,glass_training9,glass_testing9,glass_training10,glass_testing10,glass_tuning = [self.one_hot_code(df, 'class') for df in glass_list]
        # new_glass_list = [glass_training1,glass_testing1,glass_training2,glass_testing2,glass_training3,glass_testing3,glass_training4,glass_testing4,glass_training5,glass_testing5,glass_training6,glass_testing6,glass_training7,glass_testing7,glass_training8,glass_testing8,glass_training9,glass_testing9,glass_training10,glass_testing10,glass_tuning]
        # self.multi_layer_feedforward_network(len(glass_training4)-len(glass_classes), 2, 6, 6, "multiple_classification_hot", glass_training4, glass_testing4, glass_classes, 10, glass_features)

        # self.multi_layer_feedforward_network(len(soy_training1)-1, 1, 4, 4, "classification", soy_training1, soy_labels, 2)
        # soy_list = [soy_training1,soy_testing1,soy_training2,soy_testing2,soy_training3,soy_testing3,soy_training4,soy_testing4,soy_training5,soy_testing5,soy_training6,soy_testing6,soy_training7,soy_testing7,soy_training8,soy_testing8,soy_training9,soy_testing9,soy_training10,soy_testing10,soy_tuning]
        # soy_training1,soy_testing1,soy_training2,soy_testing2,soy_training3,soy_testing3,soy_training4,soy_testing4,soy_training5,soy_testing5,soy_training6,soy_testing6,soy_training7,soy_testing7,soy_training8,soy_testing8,soy_training9,soy_testing9,soy_training10,soy_testing10,soy_tuning = [self.one_hot_code(df, 'class') for df in soy_list]
        # self.multi_layer_feedforward_network(len(soy_training1)-len(soy_classes), 1, 4, 4, "multiple_classification_hot", soy_training1, soy_testing1, soy_classes, 2, soy_labels)

    def run_machine_mlps(self, num, machine_training, machine_testing, features, verbose=False):

        performance = []
        
        print("MACHINE FOLD {}, 0 layers".format(num))
        norm_train, xmax, xmin = self.regress_class_normalization(machine_training, 'erp')
        norm_test, xmax, xmin = self.regress_class_normalization(machine_testing, 'erp')
        predicted = self.multi_layer_feedforward_network(len(machine_training)-1, 0, 10, 1, "regression", norm_train, norm_test, ['erp'], 4, features, 0.01, 0.5)
        predicted = self.regress_undo_normalization(predicted,'erp',xmax,xmin)
        predicted = self.regress_undo_normalization(predicted,'Predicted',xmax,xmin)
        if verbose:
            print(predicted)
        print("LOSSES\n------------------------------------------")
        loss = self.calculate_loss_for_regression(predicted)
        performance.append(loss)
        print(loss, "\n")

        # print("\nMACHINE FOLD {}, 1 layer, 10 units, 30 epochs, 0.1 mui, 0.5 momentum".format(num))
        # predicted = self.multi_layer_feedforward_network(len(machine_training)-1, 1, 10, 1, "regression", norm_train, norm_test, ['erp'], 30, features, 0.1, 0.5)
        # predicted = self.regress_undo_normalization(predicted,'erp',xmax,xmin)
        # predicted = self.regress_undo_normalization(predicted,'Predicted',xmax,xmin)
        # if verbose:
        #     print(predicted)
        # print("LOSSES\n------------------------------------------")
        # print(self.calculate_loss_for_regression(predicted))

        print("\nMACHINE FOLD {}, 1 layer, 10 units, 70 epochs, 0.001 mui, 0.5 momentum".format(num))
        predicted = self.multi_layer_feedforward_network(len(machine_training)-1, 1, 10, 1, "regression", norm_train, norm_test, ['erp'], 70, features, 0.001, 0.5)
        predicted = self.regress_undo_normalization(predicted,'erp',xmax,xmin)
        predicted = self.regress_undo_normalization(predicted,'Predicted',xmax,xmin)
        if verbose:
            print(predicted)
        print("LOSSES\n------------------------------------------")
        loss = self.calculate_loss_for_regression(predicted)
        performance.append(loss)
        print(loss, "\n")

        # print("\nMACHINE FOLD {}, 1 layer, 10 units, 100 epochs, 0.0001 mui, 0.5 momentum".format(num))
        # predicted = self.multi_layer_feedforward_network(len(machine_training)-1, 1, 10, 1, "regression", norm_train, norm_test, ['erp'], 100, features, 0.0001, 0.5)
        # predicted = self.regress_undo_normalization(predicted,'erp',xmax,xmin)
        # predicted = self.regress_undo_normalization(predicted,'Predicted',xmax,xmin)
        # if verbose:
        #     print(predicted)
        # print("LOSSES\n------------------------------------------")
        # print(self.calculate_loss_for_regression(predicted),"\n")

        print("\nMACHINE FOLD {}, 2 layer, 10 units, 70 epochs, 0.0001 mui, 0.9 momentum".format(num))
        predicted = self.multi_layer_feedforward_network(len(norm_train)-1, 2, 10, 1, "regression", norm_train, norm_test, ['erp'], 70, features, 0.0001, 0.9)
        predicted = self.regress_undo_normalization(predicted,'erp',xmax,xmin)
        predicted = self.regress_undo_normalization(predicted,'Predicted',xmax,xmin)
        if verbose:
            print(predicted)
        print("LOSSES\n------------------------------------------")
        loss = self.calculate_loss_for_regression(predicted)
        performance.append(loss)
        print(loss, "\n\n")

        return performance

    def run_class_mlps(self, num, training, testing, features, verbose=False):
        
        pass


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
    def min_max_normalization(self, dataset: pd.DataFrame):
        df = dataset.copy()
        for col_name, col_data in df.iteritems():
            if col_name != 'class' and col_name != 'rings' and col_name != 'erp' and col_name != 'area':
                x_max = df[col_name].loc[df[col_name].idxmax()]
                # x_max = df[col_name].agg(['min', 'max'])
                x_min = df[col_name].loc[df[col_name].idxmin()]
                # df[col_name] = df[col_name].apply(lambda x: ((x - x_min)/(x_max - x_min)))
                df[col_name] = df[col_name].apply(lambda x: 2*((x - x_min)/(x_max - x_min))-1)
        return df

    # normalize numerical attributes to be in the range of -1 to +1
    # normalize numerical attributes to be in range of factor1 to factor 2
    def regress_class_normalization(self, dataset: pd.DataFrame, col_name: str, factor1=-1, factor2 = 1):
        df = dataset.copy()
        x_max = df[col_name].loc[df[col_name].idxmax()]
        # x_max = df[col_name].agg(['min', 'max'])
        x_min = df[col_name].loc[df[col_name].idxmin()]
        # df[col_name] = df[col_name].apply(lambda x: ((x - x_min)/(x_max - x_min))*factor)
        # df[col_name] = df[col_name].apply(lambda x: ((x - x_min)/(x_max - x_min)))
        df[col_name] = df[col_name].apply(lambda x: (factor2-factor1)*((x - x_min)/(x_max - x_min))+factor1)
        return df, x_max, x_min


    # normalize numerical attributes to be in the range of -1 to +1
    def regress_undo_normalization(self, dataset: pd.DataFrame, col_name: str, x_max, x_min, factor1=-1,factor2=1):
        df = dataset.copy()
        # df[col_name] = df[col_name].apply(lambda x: (x/factor) * (x_max - x_min) + x_min)
        # df[col_name] = df[col_name].apply(lambda x: x * (x_max - x_min) + x_min)
        df[col_name] = df[col_name].apply(lambda x: (x-factor1)/(factor2-factor1) * (x_max-x_min) + x_min)
        return df
        # normalize numerical attributes to be in the range of -1 to +1

    # normalize class values to be 0 or 1
    def class_normalization(self, dataset: pd.DataFrame):
        df = dataset.copy()
        x_max = df['class'].loc[df['class'].idxmax()]
        # x_max = df[col_name].agg(['min', 'max'])
        x_min = df['class'].loc[df['class'].idxmin()]
        df['class'] = df['class'].apply(lambda x: ((x - x_min)/(x_max - x_min)))
        return df

    # the weighted sum of the inputs for hidden nodes z_h 
    def sum_weight_for_hidden_nodes(self, whj, inputs, h, class_list):
        sum = 0
        # print(pd.DataFrame(inputs).T)
        # print(len(inputs)-len(class_list))
        # print(whj)
        # print(inputs)
        for j in range(len(inputs)-len(class_list)):
            # print("j:",j)
            # print("h",h)
            # print("sum",sum)
            # print(j)
            if j == 0:
                # print(1)
                # print(whj[h-1, j],"*", 1)
                sum += whj[h-1, j] * 1
                # print(sum,"+=", whj[h-1, j], "*", 1)
            else:
                # print("inputs",inputs.iat[j])
                # print(inputs[j])
                # print(whj[h-1, j-1],"*", inputs.iat[j])
                sum += whj[h-1, j-1] * inputs.iat[j]
                # print(sum,"+=", whj[h-1, j-1], "*", inputs[j])
            j += 1
            # print()
        #print("hidden", hidden)
        return sum

    # the weighted sum of the inputs for hidden nodes z_h 
    def sum_weight_for_hidden_nodes_hot(self, whj, inputs, h, num_classes):
        sum = 0
        # print(pd.DataFrame(inputs).T)
        for j in range(len(inputs)-(num_classes-1)):
            # print("sum",sum)
            # print(j)
            if j == 0:
                # print(1)
                # print(whj[h-1, j],"*", 1)
                sum += whj[h-1, j] * 1
                # print(sum,"+=", whj[h-1, j], "*", 1)
            else:
                # print(inputs[j])
                # print(whj[h-1, j-1],"*", inputs[j])
                sum += whj[h-1, j-1] * inputs[j]
                # print(sum,"+=", whj[h-1, j-1], "*", inputs[j])
            j += 1
            # print()
        #print("hidden", hidden)
        return sum

    # the activation function applied a the hidden node, sigmoid function 
    def sigmoid(self, a):
        z_h = 1 / (1 + np.exp(-a))
        return z_h
    
    # if there is just one output unit, then we computes sum weight for output node
    def sum_weight_for_output_nodes(self, vih, hiddens, i, num_hidden_units):
        # print("weight", weight)
        # print("hiddend:", hiddens)
        # print(weight)
        # print(hiddens)
        output = 0
        for h in range(0,num_hidden_units+1):
            # print("vih",vih)
            # print("zh",hiddens[h])
            output += vih[i-1, h] * hiddens[h]
            # print("output", output, "\n")
        return output

    # the update rule for classification with 2 classes
    def check_k_classes_vih(self, mui, delta, zh):
        # print("r:", r ,",", "y", y, ", mui:", mui, "," , "zh:", zh)
        delta_vh = mui * delta * zh
        return delta_vh
    
    def check_k_classes_whj(self, mui, delta, zh, vih, x, row, j_num, k, h):
        # print("r:", r ,",", "y", y, ", mui:", mui, "," , "zh:", zh, ", vh:", vh, ", x: ", row[x])
        # get sum
        sum = 0
        # print(vih)
        for i in range(1,k+1):
            # print(i,",",h)
            # print(vih[i,h])
            sum += (delta) * vih[i-1,h]
        delta_whj = mui * sum * zh * (1 - zh) * row[x]
        return delta_whj
    
    def check_k_classes_whj_hot(self, mui, class_list, y, zh, vih, x, row, j_num, k, h, l, delta_whj):
        # print("r:", r ,",", "y", y, ", mui:", mui, "," , "zh:", zh, ", vh:", vh, ", x: ", row[x])
        # get sum
        sum = 0
        # print(vih)
        if l == 0:
            for i in range(1,k+1):
                # print(i,",",h)
                # print(vih[i,h])
                sum += (row.iat[-(len(class_list)-1+i)]-y[i]) * vih[i-1,h]
        else:
            sum = delta_whj[h-1, x]
        delta_whj = mui * sum * zh * (1 - zh) * row.iat[x]
        return delta_whj


    # perform linear activation function for regression node
    def linear_activation_function(self, node):
        pass

    # perform softmax activation function for classification node
    def softmax_activaiton(self, node):
        pass

    # create multi-layer feedforward network with backpropogation 
    # Capable of training a network with arbitrary given number of inputs,
    # number of hidden layers, number of hidden units by layer, and number of outputs
    def multi_layer_feedforward_network(self, num_inputs: int, num_hidden_layers: int, num_hidden_units: int, num_outputs: int, version: str, train_df:pd.DataFrame, test_df:pd.DataFrame, class_list:list, num_iterations:int, feature_labels: list, mui, momentum, verbose=None):
        counter = 0
        # print(train_df)
        vih = np.random.uniform(-0.01, 0.01, (len(class_list), num_hidden_units+1))
        # print(vih)
        whj = np.random.uniform(-0.01, 0.01, (num_hidden_units, len(feature_labels)+1))
        # print(whj)
        # the learning rate
        mui = 0.01
        # store zh
        while(counter < num_iterations):
            train_df = train_df.copy()
            shuffed_inputs = train_df.sample(frac=1)
            hidden_dict = {}
            # store oi
            output_dict = {}
            # store yi
            yi_dict = {}
            hidden_dict[0] = 1
            # delta_vih_dict = {}
            # delta_whj_dict = {}
            # print("before", yi_dict)
            stop = 0
            for row_label, row in shuffed_inputs.iterrows():
                hidden_dict = {}
                output_dict = {}
                yi_dict = {}
                hidden_dict[0] = 1
                # delta_vih_dict = {}
                delta_vih = np.empty([len(class_list), num_hidden_units+1])
                # delta_whj_dict = {}
                delta_whj = np.empty([num_hidden_units, len(feature_labels)+1])
                self.processing_at_node(num_hidden_units, class_list,  vih, whj, hidden_dict, row, output_dict, yi_dict, version, verbose)
                # print("after oi:", output_dict)
                # print("\nafter yi:", yi_dict)
                # print("\nafter zh:", hidden_dict, "\n")
                # print("STOP:",stop)
                for l in range(num_hidden_layers):
                    if verbose == "gradient":
                        print("\nCALCULATE WEIGHT UPDATES ON LAYER {}".format(l+1))
                    if version == "binary_classification" or version == "regression":
                        # start: calculate weight updates for v_ih
                        # print(vih)
                        for h in range(0,num_hidden_units+1):
                            if l == 0:
                                # if verbose == "gradient":
                                #     print("CALCULATE NEW VIH DELTA UPDATE FOR 1ST LAYER")
                                delta_vih[0,h] = self.check_k_classes_vih(mui, row[-1] - yi_dict[1], hidden_dict[h])
                            else:
                                # if verbose == "gradient":
                                #     print("CALCULATE NEW VIH DELTA UPDATE FOR LAYER {} USING PREVIOUS LAYER".format(l+1))
                                delta_vih[0,h] = self.check_k_classes_vih(mui, delta_vih[0,h], hidden_dict[h])
                        # print("delta_vih", delta_vih)
                        # end: updating weights for v_ih
                        

                        # begin: calculate updates for w_hj
                        # print(delta_whj)
                        # print(hidden_dict)
                        # print(pd.DataFrame(row).T)
                        for h in range(1, num_hidden_units+1):
                            for j in range(len(row)-1):
                                # print(h,",",j)
                                # print(hidden_dict[h])
                                # print(row[j])
                                if l == 0:
                                    # if verbose == "gradient":
                                        # print("CALCULATE NEW WHJ DELTA UPDATE FOR 1ST LAYER")
                                    delta_whj[h-1,j] = mui * (row[-1] - yi_dict[1]) * hidden_dict[h] * (1 - hidden_dict[h]) * row[j]
                                else:
                                    # if verbose == "gradient":
                                        # print("CALCULATE NEW WHJ DELTA UPDATE FOR LAYER {} USING PREVIOUS LAYER".format(l))
                                    delta_whj[h-1,j] = mui * delta_whj[h-1,j] * hidden_dict[h] * (1 - hidden_dict[h]) * row[j]
                        # if verbose == "gradient":
                        #     print("WHJ",whj)
                        #     print("DEL_WHJ",delta_whj,"\n\n")    

                        # print(delta_whj)
                        # end: calculate updates for w_hj
                    elif version == "multiple_classification":
                        # start: calculate weight updates for v_ih
                        # print(delta_vih)
                        # print(yi_dict)
                        # print(hidden_dict)
                        for i in range(1, len(class_list)+1):
                            for h in range(0,num_hidden_units+1):
                                # if first layer gets value difference
                                if l == 0:
                                    delta_vih[i-1,h] = self.check_k_classes_vih(mui, row[-1] - yi_dict[i], hidden_dict[h])
                                # otherwise other layers get backprop?
                                else:
                                    delta_vih[i-1,h] = self.check_k_classes_vih(mui, delta_vih[i-1,h], hidden_dict[h])

                        # print("delta_vih", delta_vih)
                        # end: updating weights for v_ih

                        # begin: calculate updates for w_hj
                        for h in range(1, num_hidden_units+1):
                            for j in range(len(row)):
                                if l == 0:
                                    delta_whj[h-1,j] = self.check_k_classes_whj(mui, row[-1] - yi_dict, hidden_dict[h], vih, j, row, j, len(class_list), h)
                                else:
                                    delta_whj[h-1,j] = self.check_k_classes_whj(mui, delta_whj[h-1,j], hidden_dict[h], vih, j, row, j, len(class_list), h)
                        # print(delta_whj)
                        # end: calculate updates for w_hj
                    elif version == "multiple_classification_hot":
                        # start: calculate weight updates for v_ih
                        # print()
                        for i in range(1, len(class_list)+1):
                            for h in range(0,num_hidden_units+1):
                                if l == 0:
                                    delta_vih[i-1,h] = self.check_k_classes_vih(mui, row.iat[-(len(class_list)-1+i)] - yi_dict[i], hidden_dict[h])
                                else:
                                    delta_vih[i-1,h] = self.check_k_classes_vih(mui, delta_vih[i-1,h], hidden_dict[h])
                        # print("VIH\n",vih)
                        # print("D_VIH\n",delta_vih)
                        # end: updating weights for v_ih

                        # begin: calculate updates for w_hj
                        for h in range(1, num_hidden_units+1):
                            for j in range(len(row)-(len(class_list)-1)):
                                if l == 0:
                                    delta_whj[h-1,j] = self.check_k_classes_whj_hot(mui, class_list, yi_dict, hidden_dict[h], vih, j, row, j, len(class_list), h, l, delta_whj)
                                else:
                                    delta_whj[h-1,j] = self.check_k_classes_whj_hot(mui, class_list, yi_dict, hidden_dict[h], vih, j, row, j, len(class_list), h, l, delta_whj)
                        # print("WHJ",whj)
                        # print("DEL_WHJ",delta_whj,"\n\n")
                        # end: calculate updates for w_hj
                if verbose == "gradient":
                    print("VIH\n",vih)
                    print("D_VIH\n",delta_vih)
                    print("WHJ",whj)
                    print("DEL_WHJ",delta_whj) 
                    print("APPLY WEIGHT UPDATES FOR VIH AND WHJ\n\n".format(l+1))
                # iterative weight update with momentum
                if momentum != None:
                    if stop == 0:
                        delta_vih_t_1 = np.copy(delta_vih)
                        delta_whj_t_1 = np.copy(delta_whj)
                    vih = np.add(vih,delta_vih)
                    delta_vih_t_1 = momentum * delta_vih_t_1
                    vih = np.add(vih,delta_vih_t_1)

                    whj = np.add(whj,delta_whj)
                    delta_whj_t_1 = momentum * delta_whj_t_1
                    whj = np.add(whj,delta_whj_t_1)
                    delta_vih_t_1 = np.copy(delta_vih)
                    delta_whj_t_1 = np.copy(delta_whj)
                else:
                    # start: apply weight updates for v_ih
                    # print("VIH\n",vih)
                    # print("D_VIH\n",delta_vih)
                    # vih = np.multiply(vih,delta_vih)
                    vih = np.add(vih,delta_vih)
                    # end: updating weights for v_ih

                    # begin: apply updates for w_hj
                    # print("WHJ",whj)
                    # print("DEL_WHJ",delta_whj,"\n\n")
                    # whj = np.multiply(whj,delta_whj)
                    whj = np.add(whj,delta_whj)
                    # print(whj)
                    # end: calculate updates for w_hj
                    # stop += 1
                stop += 1
                if verbose == "gradient":
                    if stop == 5:
                        sys.exit(0)

            # end of epoch, update counter
            counter += 1
            # print(counter)
        # print("END whj")
        # print(whj)
        # print("end vih")
        # print(vih)
        # print("exited while")
        # print("\n\nRUNNING TEST\n------------------------------------------\n")
        test_df = test_df.copy()
        test_shuffed_inputs = test_df.sample(frac=1)
        if verbose == "sample_model":
            print("\nV_IH WEIGHT MATRIX: WEIGHTS FROM HIDDEN UNITS TO OUTPUT UNITS")
            print(vih)
            print("\nW_HJ WEIGHT MATRIX: WEIGHTS FROM INPUT UNITS TO HIDDEN UNITS")
            print(whj)
        for row_label, row in test_shuffed_inputs.iterrows():
            predict_yi = self.processing_at_node(num_hidden_units, class_list,  vih, whj, hidden_dict, row, output_dict, yi_dict, version, verbose)
            # print(predict_yi)
            if version == "multiple_classification" or version == "multiple_classification_hot":
                max_yi = max(predict_yi, key=predict_yi.get)
                # print("row",row_label,":    ",max_yi)
                test_shuffed_inputs.at[row_label,'Predicted'] = max_yi
            elif version == "regression":
                test_shuffed_inputs.at[row_label,'Predicted'] = predict_yi[1]

        # print("-------------------------------------------------------\n")
        # print(test_shuffed_inputs)
        return test_shuffed_inputs


    # for the nodes in network, process weight inputs to produce output
    def processing_at_node(self,
                           num_hidden_units: int, 
                           class_list:list, 
                           vih: np.matrix,
                           whj: np.matrix,
                           hidden_dict: dict,
                           row: pd.Series,
                           output_dict: dict,
                           yi_dict: dict, version: str, verbose = None):
        if verbose == "propogate":
            print(row)
        # start: hidden output weights
        # print(whj)
        # print(row)
        for h in range(1, num_hidden_units+1):
            sum_weight = self.sum_weight_for_hidden_nodes(whj, row, h, class_list)
            # print("sum:",sum_weight)
            zh = self.sigmoid(sum_weight)
            # print('zh:',zh)
            #print("zh", zh)
            hidden_dict[h] = zh
            if verbose == "propogate":
                print("zh dictionary",hidden_dict)
        # print("hidden", hidden_dict)
        # print(hidden_dict,"\n")
        # end: hidden output weights

        total = 0
        if version == "binary_classification":
            # print("BIN")
            output_dict[1] = self.sum_weight_for_output_nodes(vih, hidden_dict, 1, num_hidden_units)
            yi_dict[1] = 1 / (1 + np.exp(-output_dict[1]))
        elif version == "multiple_classification":
            # print("MULT CLASS")
            # start: output weights
            for i in range(1, len(class_list)+1):
                #print("Class",i)
                # oi = 0
                # oi += self.sum_weight_for_output_nodes(vih, hidden_dict, i-1)
                oi = self.sum_weight_for_output_nodes(vih, hidden_dict, i, num_hidden_units)
                output_dict[i] = oi
                # print("OI",oi)
                # print("exp(oi)", np.exp(oi))
                # print("Old total", total)
                total = total + np.exp(oi)
                # print("new total", total)
                # oi = 0
            # print("total: ", total)
            # end: output weights
        
            # print("Oi:",output_dict)
            # print("Total:",total)
            # start: actual output
            for i in range(1, len(class_list)+1):
                yi = np.exp(output_dict[i]) / total
                yi_dict[i] = yi
        elif version == "multiple_classification_hot":
            # print("MULT CLASS")
            # start: output weights
            for i in range(1, len(class_list)+1):
                
                oi = self.sum_weight_for_output_nodes(vih, hidden_dict, i, num_hidden_units)
                output_dict[i] = oi
                
                total = total + np.exp(oi)
                # print("new total", total)
                
            for i in range(1, len(class_list)+1):
                yi = np.exp(output_dict[i]) / total
                yi_dict[i] = yi

        elif version == "regression":
            # print("MULT CLASS")
            # start: output weights / actual output
            for i in range(1, len(class_list)+1):
                #print("Class",i)
                output_dict[1] = self.sum_weight_for_output_nodes(vih, hidden_dict, 1, num_hidden_units)
                yi_dict[1] = output_dict[1]
                if verbose == "propogate":
                    print("output", yi_dict)
                    sys.exit(0)
                # print("yi_dict", yi_dict)
            # end: output weights
                
        return yi_dict

    def calculate_loss_for_regression(self, classified_df):
        loss_dict = {}
        classified_df = classified_df.copy()
        predicted_class = classified_df["Predicted"].tolist()
        actual_class = classified_df.iloc[:,-2].tolist()
        # print(predicted_class)
        # print(actual_class)
        loss = 0
        sum, sum2, sum3 = 0, 0, 0
        all_points = []
        for i in range(len(actual_class)):
            all_points.append(math.fabs(predicted_class[i] - actual_class[i]))
            sum += math.fabs(predicted_class[i] - actual_class[i])
            sum2 += math.fabs(predicted_class[i] - actual_class[i])** 2
            sum3 += (math.fabs(predicted_class[i] - actual_class[i])/ actual_class[i]) * 100
        loss = sum / len(actual_class)
        loss_dict['MAE'] = loss
        loss_dict['MdAE'] = statistics.median(all_points)
        mse = sum2 / len(actual_class)
        loss_dict['MSE'] = mse
        mape = sum3 / len(actual_class)
        loss_dict['MAPE'] = mape
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
        return loss_dict

nn = NeuralNetwork()
nn.main()
