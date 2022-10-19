# Class to implement to KNN
from calendar import month
import math
import pandas as pd
import numpy as np
from typing import Tuple
from statistics import mode
from termcolor import colored
import sys


class KNN:
    def __init__(self):
        self.number = 7
        self.ed_flag = True
        self.kernel_flag = True
        self.knn_flag = True
        self.eknn_flag = True

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

        # get classification db classes
        cancer_classes = cancer_df['class'].unique()
        soy_classes = soy_df['class'].unique()
        glass_classes = glass_df['class'].unique()

        print("STRATIFYING DATA AND CREATING TUNING & FOLDS...")
        # Create training and testing dataframes for classification data, as well as the tuning dataframe
        cancer_training1,cancer_testing1,cancer_training2,cancer_testing2,cancer_training3,cancer_testing3,cancer_training4,cancer_testing4,cancer_training5,cancer_testing5,cancer_training6,cancer_testing6,cancer_training7,cancer_testing7,cancer_training8,cancer_testing8,cancer_training9,cancer_testing9,cancer_training10,cancer_testing10,cancer_tuning = self.stratify_and_fold_classification(cancer_df)

        # Video Step 1: Show data being split into ten folds for one data set
        # print("Glass Data")
        # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        # print(glass_df)
        # glass_training1,glass_testing1,glass_training2,glass_testing2,glass_training3,glass_testing3,glass_training4,glass_testing4,glass_training5,glass_testing5,glass_training6,glass_testing6,glass_training7,glass_testing7,glass_training8,glass_testing8,glass_training9,glass_testing9,glass_training10,glass_testing10,glass_tuning = self.stratify_and_fold_classification(glass_df)
        # print("\nGlass Ten Folds")
        # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        # print("Training Set 1")
        # print(glass_training1)
        # print("Test Set 1")
        # print(glass_testing1)
        # print("Training Set 2\n")
        # print(glass_training2)
        # print("Test Set 2")
        # print(glass_testing2)
        # print("Training Set 3")
        # print(glass_training3)
        # print("Test Set 3\n")
        # print(glass_testing3)
        # print("Training Set 4")
        # print(glass_training4)
        # print("Test Set 4\n")
        # print(glass_testing4)
        # print("Training Set 5")
        # print(glass_training5)
        # print("Test Set 5\n")
        # print(glass_testing5)
        # print("Training Set 6")
        # print(glass_training6)
        # print("Test Set 6\n")
        # print(glass_testing6)
        # print("Training Set 7")
        # print(glass_training7)
        # print("Test Set 7\n")
        # print(glass_testing7)
        # print("Training Set 8")
        # print(glass_training8)
        # print("Test Set 8\n")
        # print(glass_testing8)
        # print("Training Set 9")
        # print(glass_training9)
        # print("Test Set 9\n")
        # print(glass_testing9)
        # print("Training Set 10")
        # print(glass_training10)
        # print("Test Set 10\n")
        # print(glass_testing10)
        # print("Tuning Set")
        # print(glass_tuning,"\n\n")

        soy_training1,soy_testing1,soy_training2,soy_testing2,soy_training3,soy_testing3,soy_training4,soy_testing4,soy_training5,soy_testing5,soy_training6,soy_testing6,soy_training7,soy_testing7,soy_training8,soy_testing8,soy_training9,soy_testing9,soy_training10,soy_testing10,soy_tuning = self.stratify_and_fold_classification(soy_df)

        # Create training and testing dataframes for regression data
        abalone_training1,abalone_testing1,abalone_training2,abalone_testing2,abalone_training3,abalone_testing3,abalone_training4,abalone_testing4,abalone_training5,abalone_testing5,abalone_training6,abalone_testing6,abalone_training7,abalone_testing7,abalone_training8,abalone_testing8,abalone_training9,abalone_testing9,abalone_training10,abalone_testing10,abalone_tuning = self.stratify_and_fold_regression(abalone_df)
        machine_training1,machine_testing1,machine_training2,machine_testing2,machine_training3,machine_testing3,machine_training4,machine_testing4,machine_training5,machine_testing5,machine_training6,machine_testing6,machine_training7,machine_testing7,machine_training8,machine_testing8,machine_training9,machine_testing9,machine_training10,machine_testing10,machine_tuning = self.stratify_and_fold_regression(machine_df)
        forestfires_training1,forestfires_testing1,forestfires_training2,forestfires_testing2,forestfires_training3,forestfires_testing3,forestfires_training4,forestfires_testing4,forestfires_training5,forestfires_testing5,forestfires_training6,forestfires_testing6,forestfires_training7,forestfires_testing7,forestfires_training8,forestfires_testing8,forestfires_training9,forestfires_testing9,forestfires_training10,forestfires_testing10,forestfires_tuning = self.stratify_and_fold_regression(forestfires_df)


################################# Printing for demonstration ##############

        # print("PERFORM CLASSIFICATION USING KNN\n")
        # print("Using Glass Data Set\n")
        # glass_tune_knn1 = self.tuning_knn(glass_training1, glass_testing1, glass_tuning, 3, None, "classification")
        # print("Predicted classifications for the test instances")
        # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        # print(glass_tune_knn1,"\n")
        # self.knn_flag = True

        # print("\nPERFORM REGRESSION USING KNN\n")
        # print("Using Machine Data Set\n")
        # machine_tune_knn1 = self.tuning_knn(machine_training1, machine_testing1, machine_tuning, 3, 250, "regression")
        # print("Predicted regression for the test instances")
        # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        # print(machine_tune_knn1)
        
        # print("\nPERFORMING E-KNN ON glass SET")
        # glass_eknn_df1 = self.eknn(glass_training1,5,None,None,"classification")
        # glass_eknn1 = self.knn(glass_eknn_df1, glass_testing1, 3, None, "classification")

        # # perform kmeans using number of neighbors returned from eknn rather than tuning
        # # Perform KMEANS
        # print("\nPERFORMING KMEANS on glass DATASET")
        # glass_kmeans1 = self.km_cluster_point2(glass_training1, len(glass_eknn_df1), self.generate_centroids(glass_training1, len(glass_eknn_df1)), glass_testing1, 3, None, None, "classification", list(glass_training1))
        # get other tenfold data

        # print("#######################################################")
        # print("Get performance for classification tenfolds")
        # print("#######################################################")
        # # TUNING SETS
        # soy_tune_knn1 = self.tuning_knn(soy_training1, soy_testing1, soy_tuning, 3, None, "classification")
        # soy_tune_knn2 = self.tuning_knn(soy_training2, soy_testing2, soy_tuning, 3, None, "classification")
        # soy_tune_knn3 = self.tuning_knn(soy_training3, soy_testing3, soy_tuning, 3, None, "classification")
        # soy_tune_knn4 = self.tuning_knn(soy_training4, soy_testing4, soy_tuning, 3, None, "classification")
        # soy_tune_knn5 = self.tuning_knn(soy_training5, soy_testing5, soy_tuning, 3, None, "classification")
        # soy_tune_knn6 = self.tuning_knn(soy_training6, soy_testing6, soy_tuning, 3, None, "classification")
        # soy_tune_knn7 = self.tuning_knn(soy_training7, soy_testing7, soy_tuning, 3, None, "classification")
        # soy_tune_knn8 = self.tuning_knn(soy_training8, soy_testing8, soy_tuning, 3, None, "classification")
        # soy_tune_knn9 = self.tuning_knn(soy_training9, soy_testing9, soy_tuning, 3, None, "classification")
        # soy_tune_knn10 = self.tuning_knn(soy_training10, soy_testing10, soy_tuning, 3, None, "classification")
        # print("HAVE ALL SOY TUNINGS")
        # soy_eknn_df1 = self.eknn(soy_training1,5,None,None,"classification")
        # soy_eknn1 = self.knn(soy_eknn_df1, soy_testing1, 5, None, "classification")
        # soy_eknn_df2 = self.eknn(soy_training2,5,None,None,"classification")
        # soy_eknn2 = self.knn(soy_eknn_df2, soy_testing2, 5, None, "classification")
        # soy_eknn_df3 = self.eknn(soy_training3,5,None,None,"classification")
        # soy_eknn3 = self.knn(soy_eknn_df3, soy_testing3, 5, None, "classification")
        # soy_eknn_df4 = self.eknn(soy_training4,5,None,None,"classification")
        # soy_eknn4 = self.knn(soy_eknn_df4, soy_testing4, 5, None, "classification")
        # soy_eknn_df5 = self.eknn(soy_training5,5,None,None,"classification")
        # soy_eknn5 = self.knn(soy_eknn_df5, soy_testing5, 5, None, "classification")
        # soy_eknn_df6 = self.eknn(soy_training6,5,None,None,"classification")
        # soy_eknn6 = self.knn(soy_eknn_df6, soy_testing6, 5, None, "classification")
        # soy_eknn_df7 = self.eknn(soy_training7,5,None,None,"classification")
        # soy_eknn7 = self.knn(soy_eknn_df7, soy_testing7, 5, None, "classification")
        # soy_eknn_df8 = self.eknn(soy_training8,5,None,None,"classification")
        # soy_eknn8 = self.knn(soy_eknn_df8, soy_testing8, 5, None, "classification")
        # soy_eknn_df9 = self.eknn(soy_training9,5,None,None,"classification")
        # soy_eknn9 = self.knn(soy_eknn_df9, soy_testing9, 5, None, "classification")
        # soy_eknn_df10 = self.eknn(soy_training10,5,None,None,"classification")
        # soy_eknn10 = self.knn(soy_eknn_df10, soy_testing10, 5, None, "classification")
        # print("HAVE ALL soy EKNN")
        # # GET OTHER KMEANS 
        # soy_kmeans1 = self.km_cluster_point2(soy_training1, len(soy_eknn_df1), self.generate_centroids(soy_training1, len(soy_eknn_df1)), soy_testing1, 5, None, None, "classification", list(soy_training1))
        # soy_kmeans2 = self.km_cluster_point2(soy_training2, len(soy_eknn_df2), self.generate_centroids(soy_training2, len(soy_eknn_df2)), soy_testing2, 5, None, None, "classification", list(soy_training2))
        # soy_kmeans3 = self.km_cluster_point2(soy_training3, len(soy_eknn_df3), self.generate_centroids(soy_training3, len(soy_eknn_df3)), soy_testing3, 5, None, None, "classification", list(soy_training3))
        # soy_kmeans4 = self.km_cluster_point2(soy_training4, len(soy_eknn_df4), self.generate_centroids(soy_training4, len(soy_eknn_df4)), soy_testing4, 5, None, None, "classification", list(soy_training4))
        # soy_kmeans5 = self.km_cluster_point2(soy_training5, len(soy_eknn_df5), self.generate_centroids(soy_training5, len(soy_eknn_df5)), soy_testing5, 5, None, None, "classification", list(soy_training5))
        # soy_kmeans6 = self.km_cluster_point2(soy_training6, len(soy_eknn_df6), self.generate_centroids(soy_training6, len(soy_eknn_df6)), soy_testing6, 5, None, None, "classification", list(soy_training6))
        # soy_kmeans7 = self.km_cluster_point2(soy_training7, len(soy_eknn_df7), self.generate_centroids(soy_training7, len(soy_eknn_df7)), soy_testing7, 5, None, None, "classification", list(soy_training7))
        # soy_kmeans8 = self.km_cluster_point2(soy_training8, len(soy_eknn_df8), self.generate_centroids(soy_training8, len(soy_eknn_df8)), soy_testing8, 5, None, None, "classification", list(soy_training8))
        # soy_kmeans9 = self.km_cluster_point2(soy_training9, len(soy_eknn_df9), self.generate_centroids(soy_training9, len(soy_eknn_df9)), soy_testing9, 5, None, None, "classification", list(soy_training9))
        # soy_kmeans10 = self.km_cluster_point2(soy_training10, len(soy_eknn_df10), self.generate_centroids(soy_training10, len(soy_eknn_df10)), soy_testing10, 5, None, None, "classification", list(soy_training10))
        # print("HAVE ALL soy KMEANS")

        # print("PERFORM LOSS FUNCTIONS for Soybean")

        # soybean_loss_1 = self.calculate_loss_function(soy_tune_knn1, soy_classes, "classification")
        # print(soybean_loss_1)
        # soybean_loss_2 = self.calculate_loss_function(soy_tune_knn2, soy_classes, "classification")
        # print(soybean_loss_2)
        # soybean_loss_3 = self.calculate_loss_function(soy_tune_knn3, soy_classes, "classification")
        # print(soybean_loss_3)
        # soybean_loss_4 = self.calculate_loss_function(soy_tune_knn4, soy_classes, "classification")
        # print(soybean_loss_4)
        # soybean_loss_5 = self.calculate_loss_function(soy_tune_knn5, soy_classes, "classification")
        # print(soybean_loss_5)
        # soybean_loss_6 = self.calculate_loss_function(soy_tune_knn6, soy_classes, "classification")
        # print(soybean_loss_6)
        # soybean_loss_7 = self.calculate_loss_function(soy_tune_knn7, soy_classes, "classification")
        # print(soybean_loss_7)
        # soybean_loss_8 = self.calculate_loss_function(soy_tune_knn8, soy_classes, "classification")
        # print(soybean_loss_8)
        # soybean_loss_9 = self.calculate_loss_function(soy_tune_knn9, soy_classes, "classification")
        # print(soybean_loss_9)
        # soybean_loss_10 = self.calculate_loss_function(soy_tune_knn10, soy_classes, "classification")
        # print(soybean_loss_10)
        # soybean_loss_1 = self.calculate_loss_function(soy_tune_knn1, soy_classes, "classification")
        # sum_soybean_performance = (soybean_loss_1["Accuracy/0-1"]+soybean_loss_2["Accuracy/0-1"]+soybean_loss_3["Accuracy/0-1"]+soybean_loss_4["Accuracy/0-1"]+soybean_loss_5["Accuracy/0-1"]+soybean_loss_6["Accuracy/0-1"]+soybean_loss_7["Accuracy/0-1"]+soybean_loss_8["Accuracy/0-1"]+soybean_loss_9["Accuracy/0-1"]+soybean_loss_10["Accuracy/0-1"])/10
        # print("Average performance across classification tenfolds for k-nn:",sum_soybean_performance)

        # soybean_kmeans_loss_1 = self.calculate_loss_function(soy_kmeans1, soy_classes, "classification")
        # print(soybean_kmeans_loss_1)
        # soybean_kmeans_loss_2 = self.calculate_loss_function(soy_kmeans2, soy_classes, "classification")
        # print(soybean_kmeans_loss_2)
        # soybean_kmeans_loss_3 = self.calculate_loss_function(soy_kmeans3, soy_classes, "classification")
        # print(soybean_kmeans_loss_3)
        # soybean_kmeans_loss_4 = self.calculate_loss_function(soy_kmeans4, soy_classes, "classification")
        # print(soybean_kmeans_loss_4)
        # soybean_kmeans_loss_5 = self.calculate_loss_function(soy_kmeans5, soy_classes, "classification")
        # print(soybean_kmeans_loss_5)
        # soybean_kmeans_loss_6 = self.calculate_loss_function(soy_kmeans6, soy_classes, "classification")
        # print(soybean_kmeans_loss_6)
        # soybean_kmeans_loss_7 = self.calculate_loss_function(soy_kmeans7, soy_classes, "classification")
        # print(soybean_kmeans_loss_7)
        # soybean_kmeans_loss_8 = self.calculate_loss_function(soy_kmeans8, soy_classes, "classification")
        # print(soybean_kmeans_loss_8)
        # soybean_kmeans_loss_9 = self.calculate_loss_function(soy_kmeans9, soy_classes, "classification")
        # print(soybean_kmeans_loss_9)
        # soybean_kmeans_loss_10 = self.calculate_loss_function(soy_kmeans10, soy_classes, "classification")
        # print(soybean_kmeans_loss_10)
        # sum_soybean_kmean_performance = (soybean_kmeans_loss_1["Accuracy/0-1"]+soybean_kmeans_loss_2["Accuracy/0-1"]+soybean_kmeans_loss_3["Accuracy/0-1"]+soybean_kmeans_loss_4["Accuracy/0-1"]+soybean_kmeans_loss_5["Accuracy/0-1"]+soybean_kmeans_loss_6["Accuracy/0-1"]+soybean_kmeans_loss_7["Accuracy/0-1"]+soybean_kmeans_loss_8["Accuracy/0-1"]+soybean_kmeans_loss_9["Accuracy/0-1"]+soybean_kmeans_loss_10["Accuracy/0-1"])
        # print("Average performance across classification tenfolds for kmeans:",sum_soybean_kmean_performance)

        # soybean_eknn_loss_1 = self.calculate_loss_function(soy_eknn1, soy_classes, "classification")
        # print(soybean_eknn_loss_1)
        # soybean_eknn_loss_2 = self.calculate_loss_function(soy_eknn2, soy_classes, "classification")
        # print(soybean_eknn_loss_2)
        # soybean_eknn_loss_3 = self.calculate_loss_function(soy_eknn3, soy_classes, "classification")
        # print(soybean_eknn_loss_3)
        # soybean_eknn_loss_4 = self.calculate_loss_function(soy_eknn4, soy_classes, "classification")
        # print(soybean_eknn_loss_4)
        # soybean_eknn_loss_5 = self.calculate_loss_function(soy_eknn5, soy_classes, "classification")
        # print(soybean_eknn_loss_5)
        # soybean_eknn_loss_6 = self.calculate_loss_function(soy_eknn6, soy_classes, "classification")
        # print(soybean_eknn_loss_6)
        # soybean_eknn_loss_7 = self.calculate_loss_function(soy_eknn7, soy_classes, "classification")
        # print(soybean_eknn_loss_7)
        # soybean_eknn_loss_8 = self.calculate_loss_function(soy_eknn8, soy_classes, "classification")
        # print(soybean_eknn_loss_8)
        # soybean_eknn_loss_9 = self.calculate_loss_function(soy_eknn9, soy_classes, "classification")
        # print(soybean_eknn_loss_9)
        # soybean_eknn_loss_10 = self.calculate_loss_function(soy_eknn10, soy_classes, "classification")
        # print(soybean_eknn_loss_10)
        # sum_soybean_eknn_performance = (soybean_eknn_loss_1["Accuracy/0-1"]+soybean_eknn_loss_2["Accuracy/0-1"]+soybean_eknn_loss_3["Accuracy/0-1"]+soybean_eknn_loss_4["Accuracy/0-1"]+soybean_eknn_loss_5["Accuracy/0-1"]+soybean_eknn_loss_6["Accuracy/0-1"]+soybean_eknn_loss_7["Accuracy/0-1"]+soybean_eknn_loss_8["Accuracy/0-1"]+soybean_eknn_loss_9["Accuracy/0-1"]+soybean_eknn_loss_10["Accuracy/0-1"])
        # print("Completed all Soybean\n")

        # print("Average performance across classification tenfolds for k-nn:",sum_soybean_performance)
        # print("Average performance across classification tenfolds for ek-nn:",sum_soybean_eknn_performance)
        # print("Average performance across classification tenfolds for k-means k-nn:",sum_soybean_kmean_performance)



        print("#######################################################")
        print("Get performance for regression tenfolds")
        print("#######################################################")

        # machine_tune_knn1 = self.tuning_knn(machine_training1, machine_testing1, machine_tuning, 3, 250, "regression")
        # machine_tune_knn2 = self.tuning_knn(machine_training2, machine_testing2, machine_tuning, 3, 250, "regression")
        # machine_tune_knn3 = self.tuning_knn(machine_training3, machine_testing3, machine_tuning, 3, 250, "regression")
        # machine_tune_knn4 = self.tuning_knn(machine_training4, machine_testing4, machine_tuning, 3, 250, "regression")
        # machine_tune_knn5 = self.tuning_knn(machine_training5, machine_testing5, machine_tuning, 3, 250, "regression")
        # machine_tune_knn6 = self.tuning_knn(machine_training6, machine_testing6, machine_tuning, 3, 250, "regression")
        # machine_tune_knn7 = self.tuning_knn(machine_training7, machine_testing7, machine_tuning, 3, 250, "regression")
        # machine_tune_knn8 = self.tuning_knn(machine_training8, machine_testing8, machine_tuning, 3, 250, "regression")
        # machine_tune_knn9 = self.tuning_knn(machine_training9, machine_testing9, machine_tuning, 3, 250, "regression")
        # machine_tune_knn10 = self.tuning_knn(machine_training10, machine_testing10, machine_tuning, 3, 250, "regression")
        # print("HAVE ALL REGRESSION TUNINGS")
        # machine_tune_knn1_loss = self.calculate_loss_for_regression(machine_tune_knn1)
        # machine_tune_knn2_loss = self.calculate_loss_for_regression(machine_tune_knn2)
        # machine_tune_knn3_loss = self.calculate_loss_for_regression(machine_tune_knn3)
        # machine_tune_knn4_loss = self.calculate_loss_for_regression(machine_tune_knn4)
        # machine_tune_knn5_loss = self.calculate_loss_for_regression(machine_tune_knn5)
        # machine_tune_knn6_loss = self.calculate_loss_for_regression(machine_tune_knn6)
        # machine_tune_knn7_loss = self.calculate_loss_for_regression(machine_tune_knn7)
        # machine_tune_knn8_loss = self.calculate_loss_for_regression(machine_tune_knn8)
        # machine_tune_knn9_loss = self.calculate_loss_for_regression(machine_tune_knn9)
        # machine_tune_knn10_loss = self.calculate_loss_for_regression(machine_tune_knn10)
        # sum_machine_performance = (machine_tune_knn1_loss+machine_tune_knn2_loss+machine_tune_knn3_loss+machine_tune_knn4_loss+machine_tune_knn5_loss+machine_tune_knn6_loss+machine_tune_knn7_loss+machine_tune_knn8_loss+machine_tune_knn9_loss+machine_tune_knn10_loss)/10
        # print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        # print("Average performance across regression tenfolds for k-nn:",sum_machine_performance)

        # GET OTHER EKNNS
        machine_eknn_df1 = self.eknn(machine_training1,3, 10, 2,"regression")
        machine_eknn1 = self.knn(machine_eknn_df1, machine_testing1, 3, 500, "regression")
        print(machine_eknn1)
        machine_eknn_df2 = self.eknn(machine_training2,3,10, 2,"regression")
        machine_eknn2 = self.knn(machine_eknn_df2, machine_testing2, 3, 500, "regression")
        machine_eknn_df3 = self.eknn(machine_training3,3,10, 2,"regression")
        machine_eknn3 = self.knn(machine_eknn_df3, machine_testing3, 3, 500, "regression")
        machine_eknn_df4 = self.eknn(machine_training4,3,10, 2,"regression")
        machine_eknn4 = self.knn(machine_eknn_df4, machine_testing4, 3, 2, "regression")
        machine_eknn_df5 = self.eknn(machine_training5,3,10, 2,"regression")
        machine_eknn5 = self.knn(machine_eknn_df5, machine_testing5, 3, 2, "regression")
        machine_eknn_df6 = self.eknn(machine_training6,3,10, 2,"regression")
        machine_eknn6 = self.knn(machine_eknn_df6, machine_testing6, 3, 2, "regression")
        machine_eknn_df7 = self.eknn(machine_training7,3,10, 2,"regression")
        machine_eknn7 = self.knn(machine_eknn_df7, machine_testing7, 3, 2, "regression")
        machine_eknn_df8 = self.eknn(machine_training8,3,10, 2,"regression")
        machine_eknn8 = self.knn(machine_eknn_df8, machine_testing8, 3, 2, "regression")
        machine_eknn_df9 = self.eknn(machine_training9,3,10, 2,"regression")
        machine_eknn9 = self.knn(machine_eknn_df9, machine_testing9, 3, 2, "regression")
        machine_eknn_df10 = self.eknn(machine_training10,3,10, 2,"regression")
        machine_eknn10 = self.knn(machine_eknn_df10, machine_testing10, 3, 2, "regression")
        print("HAVE ALL MACHINE EKNN")
        machine_eknn_knn1_loss = self.calculate_loss_for_regression(machine_eknn1)
        machine_eknn_knn2_loss = self.calculate_loss_for_regression(machine_eknn2)
        machine_eknn_knn3_loss = self.calculate_loss_for_regression(machine_eknn3)
        machine_eknn_knn4_loss = self.calculate_loss_for_regression(machine_eknn4)
        machine_eknn_knn5_loss = self.calculate_loss_for_regression(machine_eknn5)
        machine_eknn_knn6_loss = self.calculate_loss_for_regression(machine_eknn6)
        machine_eknn_knn7_loss = self.calculate_loss_for_regression(machine_eknn7)
        machine_eknn_knn8_loss = self.calculate_loss_for_regression(machine_eknn8)
        machine_eknn_knn9_loss = self.calculate_loss_for_regression(machine_eknn9)
        machine_eknn_knn10_loss = self.calculate_loss_for_regression(machine_eknn10)
        sum_machine_eknn_performance = (machine_eknn_knn1_loss+machine_eknn_knn2_loss+machine_eknn_knn3_loss+machine_eknn_knn4_loss+machine_eknn_knn5_loss+machine_eknn_knn6_loss+machine_eknn_knn7_loss+machine_eknn_knn8_loss+machine_eknn_knn9_loss+machine_eknn_knn10_loss)/10
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("Average performance across regression tenfolds for ek-nn:",sum_machine_eknn_performance)


        machine_kmeans1 = self.km_cluster_point2(machine_training1, len(machine_eknn_df1), self.generate_centroids(machine_training1, len(machine_eknn_df1)), machine_testing1, 2, 10, 2,"regression", list(machine_training1))
        machine_kmeans2 = self.km_cluster_point2(machine_training2, len(machine_eknn_df2), self.generate_centroids(machine_training1, len(machine_eknn_df2)), machine_testing2, 2, 10, 2,"regression", list(machine_training2))
        machine_kmeans3 = self.km_cluster_point2(machine_training3, len(machine_eknn_df3), self.generate_centroids(machine_training2, len(machine_eknn_df3)), machine_testing3, 2, 10, 2,"regression", list(machine_training3))
        machine_kmeans4 = self.km_cluster_point2(machine_training4, len(machine_eknn_df4), self.generate_centroids(machine_training3, len(machine_eknn_df4)), machine_testing4, 2, 10, 2,"regression", list(machine_training4))
        machine_kmeans5 = self.km_cluster_point2(machine_training5, len(machine_eknn_df5), self.generate_centroids(machine_training4, len(machine_eknn_df5)), machine_testing5, 2, 10, 2,"regression", list(machine_training5))
        machine_kmeans6 = self.km_cluster_point2(machine_training6, len(machine_eknn_df6), self.generate_centroids(machine_training5, len(machine_eknn_df6)), machine_testing6, 2, 10, 2,"regression", list(machine_training6))
        machine_kmeans7 = self.km_cluster_point2(machine_training7, len(machine_eknn_df7), self.generate_centroids(machine_training6, len(machine_eknn_df7)), machine_testing7, 2, 10, 2,"regression", list(machine_training7))
        machine_kmeans8 = self.km_cluster_point2(machine_training8, len(machine_eknn_df8), self.generate_centroids(machine_training7, len(machine_eknn_df8)), machine_testing8, 2, 10, 2,"regression", list(machine_training8))
        machine_kmeans9 = self.km_cluster_point2(machine_training9, len(machine_eknn_df9), self.generate_centroids(machine_training8, len(machine_eknn_df9)), machine_testing9, 2, 10, 2,"regression", list(machine_training9))
        machine_kmeans10 = self.km_cluster_point2(machine_training10, len(machine_eknn_df10), self.generate_centroids(machine_training10, len(machine_eknn_df10)), machine_testing10, 2, 10, 2,"regression", list(machine_training10))
        print("Have all machine KMEANS")
        
        print("PERFORM LOSS FOR MACHINE")
        machine_kmeans_knn1_loss = self.calculate_loss_for_regression(machine_kmeans1)
        machine_kmeans_knn2_loss = self.calculate_loss_for_regression(machine_kmeans2)
        machine_kmeans_knn3_loss = self.calculate_loss_for_regression(machine_kmeans3)
        machine_kmeans_knn4_loss = self.calculate_loss_for_regression(machine_kmeans4)
        machine_kmeans_knn5_loss = self.calculate_loss_for_regression(machine_kmeans5)
        machine_kmeans_knn6_loss = self.calculate_loss_for_regression(machine_kmeans6)
        machine_kmeans_knn7_loss = self.calculate_loss_for_regression(machine_kmeans7)
        machine_kmeans_knn8_loss = self.calculate_loss_for_regression(machine_kmeans8)
        machine_kmeans_knn9_loss = self.calculate_loss_for_regression(machine_kmeans9)
        machine_kmeans_knn10_loss = self.calculate_loss_for_regression(machine_kmeans10)
        sum_machine_kmean_performance = (machine_kmeans_knn1_loss+machine_kmeans_knn2_loss+machine_kmeans_knn3_loss+machine_kmeans_knn4_loss+machine_kmeans_knn5_loss+machine_kmeans_knn6_loss+machine_kmeans_knn7_loss+machine_kmeans_knn8_loss+machine_kmeans_knn9_loss+machine_kmeans_knn10_loss)/10
        # print("Average performance across regression tenfolds for k-nn:",sum_machine_performance)
        print("Average performance across regression tenfolds for ek-nn:",sum_machine_eknn_performance)
        print("Average performance across regression tenfolds for k-means k-nn:",sum_machine_kmean_performance)


        

################################# KNN for Soybean Data #################################

        # soy_knn_1 = self.knn(soy_training1, soy_testing1, 3, None, "classification")
        # print(soy_knn_1)
        # print("PERFORM CLASSIFICATION USING KNN\n")
        # soy_tune_knn1 = self.tuning_knn(soy_training1, soy_testing1, soy_tuning, 3, None, "classification")
        # print("Predicted classifications for the test instances")
        # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        # print(soy_tune_knn1,"\n")
        # # perform knn using number of neighbors returned rather than tuning
        # soy_eknn_df1 = self.eknn(soy_training1,2,None,None,"classification")
        # soy_eknn1 = self.knn(soy_eknn_df1, soy_testing1, len(soy_eknn_df1), None, "classification")
        # print(soy_eknn1)
        # soy_kmeans1 = self.km_cluster_point2(soy_training1, 12, self.generate_centroids(soy_training1, 12), soy_testing1, 2, None,"classification", list(soy_training1))
        # print(soy_kmeans1)
        # soybean_loss_1 = self.calculate_loss_function(soy_tune_knn1, cancer_classes, "classification")
        # print(soybean_loss_1)

        # soy_knn = self.knn(soy_training1, soy_testing1, soy_tuning, 3, "classification")
        # print(soy_knn)
        # print("NEXT")
        # soy_knn2 = self.knn(soy_training2, soy_testing2, soy_tuning, 3, "classification")
        # print(soy_knn2)
        # print("NEXT")
        # soy_knn3 = self.knn(soy_training3, soy_testing3, soy_tuning, 3, "classification")
        # print(soy_knn3)
        # print("NEXT")


################################# KNN for Machine Data #################################

        # apply KNN to regression
        # machine_knn1 = self.knn(machine_training1, machine_testing1, 2, 250, "regression")
        # print(machine_knn1)
        # print(machine_training1)

        # machine_knn1 = self.knn(machine_training1, machine_testing1, 3, 250, "regression")
        # print(machine_knn1)
        # print("PERFORM REGRESSION USING KNN\n")
        # machine_tune_knn1 = self.tuning_knn(machine_training1, machine_testing1, machine_tuning, 3, 250, "regression")
        # print("Predicted regression for the test instances")
        # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        # print(machine_tune_knn1)
        # machine_eknn_df1 = self.eknn(soy_training1,2,None,None,"classification")
        # # perform knn using number of neighbors returned rather than tuning
        # machine_eknn = self.knn(soy_eknn_df1, soy_testing1, len(soy_eknn_df1), None, "classification")
        # print(machine_eknn)
        # machine_kmeans1 = self.km_cluster_point2(machine_training1, 10, self.generate_centroids(machine_training1, 10), machine_testing1, 2, 10, 250,"regression", list(machine_training1))
        # print(machine_kmeans1)

        # machine_kmeans1 = self.km_cluster_point2(machine_training1, 3, self.generate_centroids(machine_training1, 3), machine_testing1, 2, 1500, "regression", list(machine_training1))
        # print("1250\n",machine_kmeans1)
        # machine_kmeans1 = self.km_cluster_point2(machine_training1, 3, self.generate_centroids(machine_training1, 3), machine_testing1, 2, 1500, "regression", list(machine_training1))
        # print("1500\n",machine_kmeans1)
        # machine_kmeans1 = self.km_cluster_point2(machine_training1, 3, self.generate_centroids(machine_training1, 3), machine_testing1, 2, 1600, "regression", list(machine_training1))
        # print("1750\n",machine_kmeans1)
        # print(machine_training1)
        # machine_kmeans1 = self.km_cluster_point2(machine_training1, 3, self.generate_centroids(machine_training1, 3), machine_testing1, 2, 1500, "regression", list(machine_training1))
        # print("1250\n",machine_kmeans1)
        # machine_kmeans1 = self.km_cluster_point2(machine_training1, 3, self.generate_centroids(machine_training1, 3), machine_testing1, 2, 1500, "regression", list(machine_training1))
        # print("1500\n",machine_kmeans1)
        # machine_kmeans1 = self.km_cluster_point2(machine_training1, 3, self.generate_centroids(machine_training1, 3), machine_testing1, 2, 1600, "regression", list(machine_training1))
        # print("1750\n",machine_kmeans1)
        # machine_eknn1 = self.knn(self.eknn(machine_training1, 2, "regression"), abalone_testing1, 2, "regression")
        # print(abalone_eknn1)
        # machine_knn1 = self.knn(machine_training1, machine_testing1, machine_tuning, 2, "regression")
        # print(machine_knn1)
        # machine_knn2 = self.knn(machine_training2, machine_testing1, machine_tuning, 2, "regression")
        # print(machine_knn2)
        # machine_knn3 = self.knn(machine_training3, machine_testing1, machine_tuning, 2, "regression")
        # print(machine_knn3)
        # forestfires_knn1 = self.knn(forestfires_training1, forestfires_testing1, 2, 10, "regression")
        # print(forestfires_knn1)

################################# KNN for Forest Fire Data #################################

        # forestfires_knn = self.knn(forestfires_training1, forestfires_testing1, 3, 10, "regression")
        # print(forestfires_knn)
        # forestfires_tune_knn = self.tuning_knn(forestfires_training1, forestfires_testing1, forestfires_tuning, 3, 10, "regression")
        # print(forestfires_tune_knn)
        # forestfires_eknn = self.knn(self.eknn(forestfires_training1,2,10,1000,"regression"), forestfires_testing1, 3, 100, "regression")
        # print(forestfires_eknn)
        # forestfires_kmeans1 = self.km_cluster_point2(forestfires_training1, 10, self.generate_centroids(forestfires_training1, 10), forestfires_testing1, 3, 10,100,"regression", list(forestfires_training1))
        # print(forestfires_kmeans1)

        
        # forestfires_knn1 = self.knn(forestfires_training1, forestfires_testing1, forestfires_tuning, 2, "regression")
        # print(forestfires_knn1)
        # forestfires_knn2 = self.knn(forestfires_training2, forestfires_testing2, forestfires_tuning, 2, "regression")
        # print(forestfires_knn2)
        # forestfires_knn3 = self.knn(forestfires_training3, forestfires_testing3, forestfires_tuning, 2, "regression")
        # print(forestfires_knn3)

################################# KNN for Abalone Data #################################

        # abalone_knn1 = self.knn(abalone_training1, abalone_testing1, 2, 3, "regression")
        # print(abalone_knn1)
        # abalone_knn1 = self.knn(abalone_training1, abalone_testing1, abalone_tuning, 2, "regression")

        # abalone_knn = self.knn(abalone_training1, abalone_testing1, 3, None, "regression")
        # print(abalone_knn)
        # abalone_tune_knn = self.tuning_knn(abalone_training1, abalone_testing1, abalone_tuning, 3, None, "regression")
        # print(abalone_tune_knn)
        # abalone_eknn = self.knn(self.eknn(abalone_training1,2,10,100,"regression"), abalone_testing1, 3, None, "regression")
        # print(abalone_eknn)
        # abalone_kmeans1 = self.km_cluster_point2(abalone_training1, 12, self.generate_centroids(abalone_training1, 12), abalone_testing1, 2, None,"regression", list(abalone_training1))
        # print(abalone_kmeans1)
        
        # print(abalone_knn1)
        # abalone_knn2 = self.knn(abalone_training2, abalone_testing2, abalone_tuning, 2, "regression")
        # print(abalone_knn2)
        # abalone_knn3 = self.knn(abalone_training3, abalone_testing3, abalone_tuning, 2, "regression")
        # print(abalone_knn3)

        # Apply KNN with classification

################################# KNN for Glass Data #################################

        # glass_knn = self.knn(glass_training1, glass_testing1, glass_tuning, 5, "classification")
        # print(glass_knn)
        # print("NEXT")
        # glass_knn2 = self.knn(glass_training2, glass_testing2, glass_tuning, 5, "classification")
        # print(glass_knn2)
        # print("NEXT")
        # glass_knn3 = self.knn(glass_training3, glass_testing3, glass_tuning, 5, "classification")
        # print(glass_knn3)
        # print(glass_training1)
        # print(glass_testing1)

        # glass_knn = self.knn(glass_training1, glass_testing1, 3, None, "classification")
        # print(glass_knn)
        # glass_tune_knn = self.tuning_knn(glass_training1, glass_testing1, glass_tuning, 3, None, "classification")
        # print(glass_tune_knn)
        # glass_eknn = self.knn(self.eknn(glass_training1,2,None,None,"classification"), glass_testing1, 3, None, "classification")
        # print(glass_eknn)
        # glass_kmeans1 = self.km_cluster_point2(glass_training1, 10, self.generate_centroids(glass_training1, 10), glass_testing1, 3, None,"classification", list(glass_training1))
        # print(glass_kmeans1)

        # glass_knn = self.knn(glass_training1, glass_testing1, 2, "classification")
        # print(glass_knn)
        # print("NEXT")
        # print("NEXT")
        # glass_knn2 = self.knn(glass_training1, glass_testing1, 3, "classification")
        # print(glass_knn2)
        # print("NEXT")
        # glass_knn = self.knn(glass_training1, glass_testing1, 5, "classification")
        # print(glass_knn)
        # print("NEXT")
        # glass_knn2 = self.knn(glass_training1, glass_testing1, 10, "classification")
        # print(glass_knn2)
        # print("NEXT")
        # glass_knn3 = self.knn(glass_training1, glass_testing1, 20, "classification")
        # print(glass_knn3)
        # print("NEXT")

################################# KNN for Cancer Data #################################

        # cancer_knn = self.knn(cancer_training1, cancer_testing1, cancer_tuning, 5, "classification")
        # print(cancer_knn)
        # print("NEXT")
        # cancer_knn2 = self.knn(cancer_training2, cancer_testing2, cancer_tuning, 5, "classification")
        # print(cancer_knn2)
        # print("NEXT")
        # cancer_knn3 = self.knn(cancer_training3, cancer_testing3, cancer_tuning, 5, "classification")
        # print(cancer_knn3)
        # print(cancer_training1, cancer_testing1)

        # cancer_knn = self.knn(cancer_training1, cancer_testing1, 10, None, "classification")
        # print(cancer_knn)
        # cancer_tune_knn = self.tuning_knn(cancer_training1, cancer_testing1, cancer_tuning, 3, None, "classification")
        # print(cancer_tune_knn)
        # cancer_eknn = self.knn(self.eknn(cancer_training1,2,None,None,"classification"), glass_testing1, 3, None, "classification")
        # print(cancer_eknn)
        # cancer_kmeans1 = self.km_cluster_point2(cancer_training1, 10, self.generate_centroids(cancer_training1, 10), cancer_testing1, 10, None,"classification", list(cancer_training1))
        # print(cancer_kmeans1)

        # print("NEXT")
        # cancer_knn = self.knn(cancer_training1, cancer_testing1, 2, None,"classification")
        # print(cancer_knn)
        # print("NEXT")
        # cancer_knn = self.knn(cancer_training1, cancer_testing1, 3, "classification")
        # print(cancer_knn)
        # print("NEXT")
        # cancer_knn = self.knn(cancer_training1, cancer_testing1, 5, "classification")
        # print(cancer_knn)
        # print("NEXT")
        # cancer_knn = self.knn(cancer_training1, cancer_testing1, 10, "classification")
        # print(cancer_knn)
        # print("NEXT")

        # apply KMEANS
        # cancer_kmeans1 = self.km_cluster_point2(cancer_training1, 3, self.generate_centroids(cancer_training1, 3), cancer_testing1, 5, "classification", list(cancer_training1))
        # print(cancer_kmeans1)
        # machine_kmeans1 = self.km_cluster_point2(machine_training1, 3, self.generate_centroids(machine_training1, 3), machine_testing1, 5, "regression", list(machine_training1))
        # print(machine_kmeans1)

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
        dataset =dataset.reindex(columns=["F","I","M","diameter","height","whole_weight","shucked_weight","viscera_weight","shell_weight","rings"])
        # print(dataset)
        return dataset

    # function for forest fires cyclical ordinals
    def cyclical_ordinals(self, df: pd.DataFrame):
        print("Entering Cyc")
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

    # function to implement creation of value difference matrices for a given training set
    def value_difference_metric(self, train_df: pd.DataFrame) -> None:
        train_df = train_df.copy()
        print("Entering VDM...")
        # empty dictionary to hold all feature difference matrices
        diff_matrix_dict = {}
        classes = train_df.iloc[:,-1].unique()
        # for each feature in x
        # print(train_df)
        for col_name, col_value in train_df.iteritems():
            if col_name != "class":
                # print(col_name)
                # print(col_value)
                # array of possible values for that feature
                values = col_value.unique()
                # create empty feature differences matrix
                diff_matrix = np.empty((len(values),len(values)))
                # construct feature differences matrix
                for idx_i, vi in enumerate(values):
                    for idx_j, vj in enumerate(values):
                        # calculate d(vi,vj) sum over classes
                        d_vi_vj = 0
                        for c in classes:
                            ci = train_df[train_df[col_name] == vi].shape[0]
                            cia = train_df[(train_df[col_name] == vi) & (train_df.iloc[:,-1] == c)].shape[0]
                            cj = train_df[train_df[col_name] == vj].shape[0]
                            cja = train_df[(train_df[col_name] == vj) & (train_df.iloc[:,-1] == c)].shape[0]
                            d_vi_vj += pow(abs((cia/ci) - (cja/cj)),2)
                        # put d(vi,vj) in the feature differences matrix
                        diff_matrix[idx_i,idx_j] = d_vi_vj
                # put the feature valu matrix into the dictionary
                diff_matrix_dict[col_name] = diff_matrix
        # return the dictionary of all feature difference matrices
        return diff_matrix_dict
    
    # Function to make a regression prediction using Guassian Kernel
    def guassian_kernel(self, train_df: pd.DataFrame, test_row: pd.Series, k:int, sigma) -> None:
        if self.kernel_flag:
            print("Entering Kernel Function")
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print("Training df")
            print(train_df,"\n")
            print("Test instance")
            print(test_row)
        train_df = train_df.copy()
        near_ks = train_df.iloc[k:]
        sum_up = 0
        sum_down = 0 
        cnt = 0
        for r_index, train_row in near_ks.iterrows():
            ed = train_row["Distance"]
            if self.kernel_flag and cnt < 4:
                print("rt:",train_row[-2])
            # caluclate kernal equation
            try:
                kernal = math.exp(-(math.pow(ed, 2) / (2 * math.pow(sigma, 2))))
            except TypeError:
                kernal = math.exp(-(math.pow(0, 2) / (2 * math.pow(sigma, 2))))
            if self.kernel_flag and cnt < 4:
                print("dist squared",math.pow(ed, 2))
                print("divide by this",(2 * math.pow(sigma, 2)))
                print("before exp",-(math.pow(ed, 2) / (2 * math.pow(sigma, 2))))
                print("\nK(xq,xt) = ",kernal)
            sum_up += kernal * train_row[-2]
            sum_down += kernal
            if self.kernel_flag and cnt < 4:
                print("\nNew top sum",sum_up)
                print("New bottom sum",sum_down)
                cnt += 1
                print("The rest of the summations will not be displayed to save space")
        sum = sum_up / sum_down
        if self.kernel_flag:
            print("\ng(x) =",sum)
            self.kernel_flag = False
        return sum
    
    # function to perform knn on given training and test datasets
    def tuning_knn(self, train_df: pd.DataFrame, test_df: pd.DataFrame, tune_df: pd.DataFrame, k_start: int, sigma: int, version: str) -> pd.DataFrame:
        train_df = train_df.copy()
        train_df3 = train_df.copy()
        test_df = test_df.copy()
        tune_df = tune_df.copy()
        if version == "classification":
            print("Entering Tuning KNN for Classification...")
        else:
            print("Entering Tuning KNN for Regression")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        k_tune_values = []
        if version == "classification":
            # get feature difference matrices
            diff_matrix_dict = self.value_difference_metric(train_df)
            # print(diff_matrix_dict)
        # Tune K by trying out five possible values around the base k value and choosing the k with best predictions
        for k in range(max(0,k_start-2),k_start+3):
            train_df2 = train_df.copy()
            tune_df2 = tune_df.copy()
            # print("ROUND w/ K =", k)
            predictions = []
            # Loop through each instance in the tuning dataset
            for tune_row_index, tune_row in tune_df2.iterrows():
                distances = []
                for train_row_index, train_row in train_df2.iterrows():
                    # apply Euclidean distance funciton if regression
                    if version == "regression":
                        # Get euclidean distance between current tune instance and a given instance in tune set
                        distances.append(self.euclidean_distance(train_row, tune_row))
                    # apply distance based on value difference metric if classification
                    elif version == "classification":
                        d_x_y = 0
                        col_names = list(train_row.index)
                        for name in col_names:
                            if name != "class" and name != "Distance":
                                try:
                                    d_x_y += diff_matrix_dict[name][train_row[name]][tune_row[name]]
                                except IndexError:
                                    d_x_y += 0
                        # Add distance between current tune instance and a given instance in tune set to distances array
                        distances.append(math.sqrt(d_x_y))
                # Add the returned distances onto the end of the training set
                train_df["Distance"] = distances
                # Find the min k distances in the training set
                sorted_df = train_df.sort_values("Distance").reset_index(drop=True).copy()
                # Predict the mean of the k smallest distance instances
                if version == "regression":
                    k_sum = 0
                    for i in range(k):
                        k_sum += sorted_df.iat[i,-2]
                    predictions.append(k_sum/k)
                # Predict the most occuring class of the k smallest distance instances
                elif version == "classification":
                    k_classes = []
                    for i in range(k):
                        # k_classes = sorted_df.loc[sorted_df.index[i], "class"]
                        k_classes.append(sorted_df.at[i, "class"])
                    predictions.append(mode(k_classes))
                # Move to the next tune instance
            tune_df['KNN_Prediction'] = predictions
            # print("WITH",k,"NEIGHBORS")
            # print(tune_df)
            prediction_cnt = 0
            # Check accuracy of predicitions
            for row_label, row in tune_df.iterrows():
                # print(row)
                # print(row["KNN_Prediction"])
                # print(row[row.index[-2]])
                if row["KNN_Prediction"] == row[row.index[-2]]:
                    prediction_cnt += 1
            # print(prediction_cnt)
            # print(prediction_cnt/tune_df.shape[0])
            k_tune_values.append((k,prediction_cnt/tune_df.shape[0]))
            # sys.exit(0)
        # set k to the value with best performance
        # print(k_tune_values)
        k = max(k_tune_values, key=lambda tup: tup[1])[0]
        print("Tuned k to be:", k,"\n")

        # Use chosen k value on test data
        result_df = self.knn(train_df3,test_df,k,sigma,version)
        return result_df

    def knn(self, train_df: pd.DataFrame, test_df:pd.DataFrame, k: int, sigma: int, version: str) -> pd.DataFrame:
        if self.knn_flag:
            print("Entering Basic KNN...")
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        train_df = train_df.copy()
        train_df2 = train_df.copy()
        # print(train_df)
        test_df = test_df.copy()
        predictions = []
        # print(train_df)
        # print(test_df)
        if version == "classification":
            # get feature difference matrices
            diff_matrix_dict = self.value_difference_metric(train_df)
        # Loop through each instance in the testing dataset
        for test_row_index, test_row in test_df.iterrows():
            # print("TESTROW STAERT")
            # Loop through each instance in the training set
            distances = []
            for train_row_index, train_row in train_df2.iterrows():
                # print("ITERESTART")
                # print(train_df)
                # apply Euclidean distance funciton if regression
                if version == "regression":
                    # Get euclidean distance between current test instance and a given instance in test set
                    distances.append(self.euclidean_distance(train_row, test_row))
                # apply distance based on value difference metric if classification
                elif version == "classification":
                    d_x_y = 0
                    col_names = list(train_row.index)
                    # print(diff_matrix_dict)
                    for name in col_names:
                        # print(name,":")
                        if name != "class" and name != "Distance":
                            try:
                                d_x_y += diff_matrix_dict[name][train_row[name]][test_row[name]]
                            except IndexError:
                                d_x_y += 0
                    # Add distance between current test instance and a given instance in test set to distances array
                    distances.append(math.sqrt(d_x_y))
            # print("OUT")
            # Add the returned distances onto the end of the training set
            train_df["Distance"] = distances
            # Find the min k distances in the training set
            sorted_df = train_df.sort_values("Distance").reset_index(drop=True).copy()
            # Show the neighbors returned
            if self.knn_flag:
                print("The K Neighbors for instance 1")
                print("+++++++++++++++++++++++++++")
                print(sorted_df.head(k))
                self.knn_flag = False
            # Predict the mean of the k smallest distance instances
            if version == "regression":
                k_sum = self.guassian_kernel(sorted_df, test_row, k, sigma)
                predictions.append(k_sum/k)
            # Predict the most occuring class of the k smallest distance instances
            elif version == "classification":
                k_classes = []
                for i in range(min(k,len(sorted_df))):
                    # k_classes = sorted_df.loc[sorted_df.index[i], "class"]
                    k_classes.append(sorted_df.at[i, "class"])
                predictions.append(mode(k_classes))
                # predictions.append("D1")
            # Move to the next test instance
        # Set the predictions to a column on the test data set
        test_df['KNN_Prediction'] = predictions
        # Return the test set with KNN Predictions appended
        return test_df

    # function to calculate the euclidean distance between a training instance and a test instance
    def euclidean_distance(self, train_row: pd.Series, test_row: pd.Series) -> int:
        # Video Req 2: Show calculation of distance function for first call of euclidean distance
        if self.ed_flag:
            print("Entering the Euclidean Distance function")
            print("++++++++++++++++++++++++++++++++++++++++")
            print("Training instance")
            print(train_row,"\n")
            print("Test instance")
            print(test_row)
        euclidean_distance = 0
        # Get sum of attribute distances
        for i in range(0,train_row.shape[0]-1):
            if self.ed_flag:
                print("Euclidean Distance Sum:",euclidean_distance)
            euclidean_distance += pow((train_row[i] - test_row[i]),2)
        if self.ed_flag:
            print("Calculated square root Euclidean Dist:",math.sqrt(euclidean_distance))
            self.ed_flag = False
        # Return the square root of the sum
        return math.sqrt(euclidean_distance)

    # function to perform edited k nearest neighbor on a given training dataset, using error removal method
    # returns reduced dataset
    def eknn(self, dataset: pd.DataFrame, k: int, epsilon: int, sigma:int, version: str) -> pd.DataFrame:
        # run until points are not edited
        print("Entering Edited-KNN")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        dataset = dataset.copy()
        performance_flag = True
        dataset = dataset.head(1500)
        print("ORIGINAL TRAINING SET")
        print(dataset)
        while performance_flag:
            remove_points = []
            cnt = 0
            original_size = dataset.shape[0]
            indx = 1
            # loop through all points in E
            # print("Length",original_size)
            for row_index, row in dataset.iterrows():
                # Classify xi using knn w/ all other points in E
                df_no_xi = dataset.drop(row_index)
                # print(df_no_xi)
                # print()
                # print(row)
                # print()
                # row_df = row.to_frame().reset_index().T
                row_df = pd.DataFrame(row).transpose()
                # print(row_df)
                predicted_row = self.knn(df_no_xi, row_df, k, sigma, version)
                # Compare the classification returned with the stored class label
                if version == 'classification':
                    if predicted_row.iat[0,-1] != row[-1]:
                        remove_points.append(row_index)
                        print("Row",indx,"is done")
                    else:
                        cnt += 1
                    print("Row",indx,"is done")
                    indx += 1
                # Compare with error threshold for regression prediction
                if version == 'regression':
                    # If the prediction difference is greater than the error threshhold.
                    if abs(predicted_row.iat[0,-1] - row[-1]) > epsilon:
                        remove_points.append(row_index)
                    else:
                        cnt += 1
                    print("Row",indx,"is done")
                    indx += 1
            # remove all points from E where the classifaction was incorrect
            print(remove_points)
            dataset.drop(remove_points, inplace=True)
            # if no points removed from E this round, end loop and return
            if cnt == original_size:
                performance_flag = False
                # print("FINITO")
                edited_dataset = dataset.copy()
            print("Points to be removed")
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print("Previous dataset length:",original_size,"--------- New_Dataset:",cnt)
            print(dataset)
        # print(edited_dataset)
        print("^^^ THis is the final reduced dataset of size",edited_dataset.shape,"\n\n")
        return edited_dataset

    # step to generate the centroids randomly
    def generate_centroids(self, data: pd.DataFrame, k: int):
        centroids = data.iloc[np.random.choice(np.arange(len(data)), k, False)]
        return centroids

    def km_cluster_point2(self, sub_data: pd.DataFrame, k: int, centroids: pd.DataFrame, test_df: pd.DataFrame, k_knn: int, epsilon: int, sigma: int, version: str,labels: list) -> None:
        print("INTO CLUSTERING")
        # print(sub_data)
        test_df = test_df.copy()
        stop = 1
        flag = True
        self.eknn_flag = True
        # print(centroids)
        # old_centeroids =  centroids.reset_index(drop=True)
        old_centeroids =  centroids.copy()
        print("Old Centeroids")
        print(old_centeroids)
        if version == "classification":
            # get feature difference matrices
            diff_matrix_dict = self.value_difference_metric(sub_data)
        while (stop != 0):
            # print(old_centeroids)
            # to store the distance 
            temp_data = sub_data.copy()
            index = 1
            if version == "regression":
                for c_row_idx, c_row in old_centeroids.iterrows():
                    dists = []
                    for d_row_idx, d_row in sub_data.iterrows():
                        ed = self.euclidean_distance(c_row, d_row)
                        dists.append(ed)
                    temp_data[index] = dists
                    index = index + 1
            elif version == "classification":
                for c_row_idx, c_row in enumerate(old_centeroids.itertuples(index=False)):
                    dists = []
                    for d_row_idx, d_row in enumerate(sub_data.itertuples(index=False)):
                        d_x_y = 0
                        col_names = list(sub_data.columns)
                        for name in col_names:
                            if name != "class":
                                try:
                                    # d_x_y += diff_matrix_dict[name][d_row[name]][c_row[name]]
                                    d_x_y += diff_matrix_dict[name][d_row[sub_data.columns.get_loc(name)]][c_row[old_centeroids.columns.get_loc(name)]]
                                except IndexError:
                                    d_x_y += 0
                        dists.append(math.sqrt(d_x_y))
                    temp_data[index] = dists
                    index = index + 1
            # if self.eknn_flag:
            #     #print("\nData points are assigned distances from each centroid")
            #     print(temp_data)
            # sys.exit(0)
            # create a list to store clusters 
            cluster = []
            # for all points in dataset, assign to a cluster
            # print(temp_data)
            for ed_index, ed_row in temp_data.iterrows():
                # set the first position of the row to be the min
                min_distance = ed_row[1]
                cluster_num = 1
                for i in range(0,k):
                    try:
                        if ed_row[i+1] < min_distance:
                            cluster_num = i + 1
                    except KeyError:
                        pass
                cluster.append(cluster_num)
            temp_data["Cluster"] = cluster
            if self.eknn_flag:
                print("Distances are used to assign data to appropriate Cluster")
                print(temp_data)
                self.eknn_flag = False
            # print(temp_data)
            cluster_groups = temp_data.groupby(['Cluster'])
            i = 0
            distortions = []
            centeroid_idx = list(old_centeroids.index.values)
            old_centeroids["Distortion"] = [0] * len(centeroid_idx)
            # for each medoid, get its distortion
            for index, centeroid in cluster_groups:
                distortion_sum = 0
                # for each data point in that medoid's cluster
                for row_index, cluster_row in centeroid.iterrows():
                    distortion_sum += cluster_row[i+1]
                    # distortion_sum += self.euclidean_distance(c_row, d_row)
                # distortions.append(distortion_sum)
                old_centeroids.at[centeroid_idx[i],"Distortion"]
                i += 1
            # old_centeroids["Distortion"] = distortions
            # print(old_centeroids)
            # sys.exit(0)
            i = 0
            new_centeroids = [None] * len(centeroid_idx)
            new_temp_data = temp_data.copy()
            # for each cluster group
            for index, centeroid in cluster_groups:
                new_distorts = []
                if i+1 != index:
                    new_centeroids[i] = old_centeroids.iloc[i]
                    i += 1
                # for each data point in the cluster
                # print(centeroid)
                # for row_index, cluster_row in centeroid.iterrows():
                # print(centeroid)
                if version == "regression":
                    for row_index, cluster_row in centeroid.iterrows():
                        if not pd.Series.equals(cluster_row,old_centeroids.iloc[i]):
                            # get distortion of data point
                            new_distortion = 0
                            for d_row_idx, c_row in centeroid.iterrows():
                                new_distortion += self.euclidean_distance(cluster_row[0:-(k+2)], c_row[0:-(k+2)])
                            new_distorts.append(new_distortion)
                elif version == "classification":
                    for row_index, cluster_row in enumerate(centeroid.itertuples(index=False)):
                        # if the data point is not a medoid
                        if not pd.Series.equals(cluster_row,old_centeroids.iloc[i]):
                            # get distortion of data point
                            new_distortion = 0
                            # for d_row_idx, d_row in centeroid.iterrows():
                            for d_row_idx, d_row in enumerate(centeroid.itertuples(index=False)):
                                d_x_y = 0
                                col_names = list(sub_data.columns)
                                for name in col_names:
                                    if name != "class":
                                        try:
                                            # d_x_y += diff_matrix_dict[name][d_row[name]][cluster_row[name]]
                                            d_x_y += diff_matrix_dict[name][d_row[centeroid.columns.get_loc(name)]][cluster_row[centeroid.columns.get_loc(name)]]
                                        except IndexError:
                                            d_x_y += 0
                                new_distortion += math.sqrt(d_x_y)
                            new_distorts.append(new_distortion)
                # print("Distortions",new_distorts)
                centeroid["Distortion"] = new_distorts
                min_cent = centeroid[centeroid["Distortion"] == centeroid["Distortion"].min()]
                min_whole = min_cent.iloc[0,0:-5]
                min_whole["Distortion"] = min_cent.iloc[0,-1]
                if centeroid["Distortion"].min() < old_centeroids.iloc[i]["Distortion"]:
                    new_centeroids[i] = min_whole
                    # print("New centroid had smaller distortion!")
                else:
                    new_centeroids[i] = old_centeroids.iloc[i]
                    # print("Keep old one!")
                i += 1
            new_centeroids_df = pd.concat(new_centeroids, axis=1).transpose()
            if version == "classification":
                for col in new_centeroids_df.columns:
                    if col != "class" and col != "Distortion":
                        new_centeroids_df[col] = new_centeroids_df[col].astype(int)
            # print("OLDS")
            # print(old_centeroids)
            # print("NEWS")
            # print(new_centeroids_df)
            # print("Is same points?",pd.DataFrame.equals(old_centeroids,new_centeroids_df))
            # print()
            # if flag:
                # print("OLDS")
                # print(old_centeroids)
                # print("NEWS")
                # print(new_centeroids_df)
                # print("Is same points?",pd.DataFrame.equals(old_centeroids,new_centeroids_df))
                # print()
            #     flag = False
            # print(new_centeroids_df)
            if pd.Index.equals(old_centeroids.index,new_centeroids_df.index):
                # print("OLDS")
                # print(old_centeroids)
                # print("NEWS")
                # print(new_centeroids_df)
                # print("Is same points?",pd.DataFrame.equals(old_centeroids,new_centeroids_df))
                # print()
                stop = 0
            # names = list(new_centeroids_df)
            # new_centeroids[names[0:-3]] = new_centeroids_df[names[0:-3]].astype(int)
            # print("W/ ints")
            old_centeroids = new_centeroids_df
        # print("Finished!")
        new_centeroids_df = new_centeroids_df.drop("Distortion", axis=1)
        print("\nFinal Chosen centeroids")
        print(new_centeroids_df)
        # Use the medoids as training set in knn
        return self.knn(new_centeroids_df,test_df,k_knn,sigma,version)

    #function to create confusion matrix
    def calculate_loss_function(self, classified_df, class_names, version):
        classified_df = classified_df.copy()
        confusion_matrix = {}
        actual_class = classified_df["class"].tolist()
        if version == "classification":
            predicted_class = classified_df["KNN_Prediction"].tolist()
            confusion_matrix = self.get_predicted_class(confusion_matrix, class_names, actual_class, predicted_class)
        elif version == "regression":
            predicted_class = classified_df["KNN_Prediction"].tolist()
            confusion_matrix = self.get_predicted_class(confusion_matrix, class_names, actual_class, predicted_class)
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
        F1 = 2 * ((precision * recall) / (precision + recall))
        accuracy = (total_tp + total_tn) / total
        loss = {"Accuracy/0-1": accuracy, "Precision": precision, "Recall": recall, "F1": F1}
        return loss
    def get_predicted_class(self, confusion_matrix, class_names, actual_class, predicted_class):
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
    def calculate_loss_for_regression(self, classified_df):
        sigma = 0.5
        classified_df = classified_df.copy()
        predicted_class = classified_df["KNN_Prediction"].tolist()
        actual_class = classified_df.iloc[:,-2].tolist()
        print(predicted_class)
        print(actual_class)
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
        print(loss_list)
        return loss

    
knn = KNN()
knn.main()