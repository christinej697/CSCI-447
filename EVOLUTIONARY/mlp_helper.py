from mlp import MLP
from utils import UTILS
import pandas as pd
from random import random
import numpy as np

class MLP_HELPER:
    def __init__(self):
        pass

    ######### Regression Data ##################################
    def mlp_machine_data():
        machine_labels = ["vendor_name","model","myct","mmin","mmax","cach","chmin","chmax","prp","erp"]
        machine_df = UTILS.import_data(UTILS, "machine.data",machine_labels)
        machine_df = machine_df.drop(['vendor_name','model'], axis = 1)
        new_machine_df = UTILS.min_max_normalization(UTILS, machine_df)
        machine_training1,machine_testing1,machine_training2,machine_testing2,machine_training3,machine_testing3,machine_training4,machine_testing4,machine_training5,machine_testing5,machine_training6,machine_testing6,machine_training7,machine_testing7,machine_training8,machine_testing8,machine_training9,machine_testing9,machine_training10,machine_testing10,machine_tuning = UTILS.stratify_and_fold_regression(UTILS, new_machine_df)
        test_list = [machine_testing1,machine_testing2,machine_testing3,machine_testing4,machine_testing5,machine_testing6,machine_testing7,machine_testing8,machine_testing9,machine_testing10]
        return test_list

    def mlp_abalone_data():
        abalone_labels = ["sex","length","diameter","height","whole_weight","shucked_weight","viscera_weight","shell_weight","rings"]
        abalone_df = UTILS.import_data(UTILS, "abalone.data",abalone_labels)
        abalone_df = UTILS.one_hot_code(UTILS, abalone_df, 'sex')
        new_abalone_df = UTILS.min_max_normalization(UTILS, abalone_df)
        abalone_training1,abalone_testing1,abalone_training2,abalone_testing2,abalone_training3,abalone_testing3,abalone_training4,abalone_testing4,abalone_training5,abalone_testing5,abalone_training6,abalone_testing6,abalone_training7,abalone_testing7,abalone_training8,abalone_testing8,abalone_training9,abalone_testing9,abalone_training10,abalone_testing10,abalone_tuning = UTILS.stratify_and_fold_regression(UTILS, new_abalone_df)
        test_list=[abalone_testing1, abalone_testing2, abalone_testing3, abalone_testing4, abalone_testing5, abalone_testing6, abalone_testing7, abalone_testing8, abalone_testing9, abalone_testing10]
        return test_list
    
    def mlp_forestfires_data():
        forestfires_df = pd.read_csv("forestfires.csv", sep=",")
        forestfires_df = UTILS.cyclical_ordinals(UTILS, forestfires_df)
        new_forestfires_df = UTILS.min_max_normalization(forestfires_df)
        forestfires_training1,forestfires_testing1,forestfires_training2,forestfires_testing2,forestfires_training3,forestfires_testing3,forestfires_training4,forestfires_testing4,forestfires_training5,forestfires_testing5,forestfires_training6,forestfires_testing6,forestfires_training7,forestfires_testing7,forestfires_training8,forestfires_testing8,forestfires_training9,forestfires_testing9,forestfires_training10,forestfires_testing10,forestfires_tuning = UTILS.stratify_and_fold_regression(UTILS, new_forestfires_df)
        test_list = [forestfires_testing1, forestfires_testing2, forestfires_testing3, forestfires_testing4, forestfires_testing5, forestfires_testing6, forestfires_testing7, forestfires_testing8, forestfires_testing9, forestfires_testing10]
        return test_list

    ###############################################################
    ######### Classification Data ##################################
    def mlp_glass_data():
        glass_labels = ['id-num','retractive-index','sodium','magnesium','aluminum','silicon','potasium','calcium','barium','iron','class']
        glass_df = UTILS.import_data(UTILS, "glass.data", glass_labels)
        glass_df.drop(columns=glass_df.columns[0], axis=1, inplace=True)
        new_glass_df = UTILS.min_max_normalization(UTILS, glass_df)
        glass_training1,glass_testing1,glass_training2,glass_testing2,glass_training3,glass_testing3,glass_training4,glass_testing4,glass_training5,glass_testing5,glass_training6,glass_testing6,glass_training7,glass_testing7,glass_training8,glass_testing8,glass_training9,glass_testing9,glass_training10,glass_testing10,glass_tuning = UTILS.stratify_and_fold_classification(UTILS, new_glass_df)
        test_list = [glass_testing1, glass_testing2, glass_testing3, glass_testing4, glass_testing5, glass_testing6, glass_testing7, glass_testing8, glass_testing9, glass_testing10]
        classes = [1, 2, 3, 4, 5, 6, 7]
        return test_list

    def mlp_cancer_data(cancer_mlp, learning_rate, iterations):
        cancer_labels = ["Sample code number", "Uniformity of Clump Thickness", "Uniformity of Cell Size", "Cell Shape", "Marginal Adhesion", "Single Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses", "class"]
        cancer_df = UTILS.import_data(UTILS,"breast-cancer-wisconsin-cleaned.txt", cancer_labels)
        cancer_df.drop(columns=cancer_df.columns[0], axis=1, inplace=True)
        new_cancer_df = UTILS.min_max_normalization(UTILS, cancer_df)
        cancer_training1,cancer_testing1,cancer_training2,cancer_testing2,cancer_training3,cancer_testing3,cancer_training4,cancer_testing4,cancer_training5,cancer_testing5,cancer_training6,cancer_testing6,cancer_training7,cancer_testing7,cancer_training8,cancer_testing8,cancer_training9,cancer_testing9,cancer_training10,cancer_testing10,cancer_tuning = UTILS.stratify_and_fold_classification(UTILS, new_cancer_df)
        test_list = [cancer_testing1, cancer_testing2, cancer_testing3, cancer_testing4, cancer_testing5, cancer_testing6, cancer_testing7, cancer_testing8, cancer_testing9, cancer_testing10]
        return test_list

    def mlp_soybean_data(soybean_mlp, learning_rate, iterations):
        soy_labels = ["date","plant-stand","precip","temp","hail","crop-hist","area-damaged","severity","seed-tmt","germination","leaves","lodging","stem-cankers","canker-lesion","fruiting-bodies","external-decay","mycelium","int-discolor","sclerotia","fruit-pods","roots","class"]
        soy_df = UTILS.import_data(UTILS,"soybean-small-cleaned.csv", soy_labels)
        classes = soy_df["class"]
        UTILS.one_hot_code(UTILS, classes)
        new_soybean_df = UTILS.min_max_normalization(UTILS, soy_df)
        soy_training1,soy_testing1,soy_training2,soy_testing2,soy_training3,soy_testing3,soy_training4,soy_testing4,soy_training5,soy_testing5,soy_training6,soy_testing6,soy_training7,soy_testing7,soy_training8,soy_testing8,soy_training9,soy_testing9,soy_training10,soy_testing10,soy_tuning = UTILS.stratify_and_fold_classification(UTILS, new_soybean_df)
        test_list = [soy_testing1, soy_testing2, soy_testing3, soy_testing4, soy_testing5, soy_testing6, soy_testing7, soy_testing8, soy_testing9, soy_testing10]
        classes = [1, 2, 3, 4]
        return test_list
            
    def get_mlp_weights(mlp, test_list):
        test_output_list = []
        for i in range(len(test_list)):
            testing_np = test_list[i].to_numpy()
            mlp.forward_feed(testing_np)
            weights = mlp.get_weights()
            test_output_list.append(weights)
        return test_list

