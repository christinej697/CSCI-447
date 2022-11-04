from mlp import MLP
from utils import UTILS
import pandas as pd
from random import random

def mlp_glass_data(glass_mlp, learning_rate, iterations):
    glass_labels = ['id-num','retractive-index','sodium','magnesium','aluminum','silicon','potasium','calcium','barium','iron','class']
    glass_df = UTILS.import_data(UTILS, "glass.data", glass_labels)
    glass_df.drop(columns=glass_df.columns[0], axis=1, inplace=True)
    print()
    print("glass dataframe: ")
    print(glass_df)

    new_glass_df = UTILS.min_max_normalization(UTILS, glass_df)
    print()
    print("normalized glass dataframe: ")
    print(new_glass_df)

    glass_training1,glass_testing1,glass_training2,glass_testing2,glass_training3,glass_testing3,glass_training4,glass_testing4,glass_training5,glass_testing5,glass_training6,glass_testing6,glass_training7,glass_testing7,glass_training8,glass_testing8,glass_training9,glass_testing9,glass_training10,glass_testing10,glass_tuning = UTILS.stratify_and_fold_classification(UTILS, new_glass_df)
    train_list = [glass_training1, glass_training2, glass_training3, glass_training4, glass_training5, glass_training6, glass_training7, glass_training8, glass_training9, glass_training10]
    test_list = [glass_testing1, glass_testing2, glass_testing3, glass_testing4, glass_testing5, glass_testing6, glass_testing7, glass_testing8, glass_testing9, glass_testing10]
    classes = [1, 2, 3, 4, 5, 6, 7]
    performance = data_processing(train_list, test_list, glass_mlp, learning_rate, iterations, classes)
    print()
    print("Final resuld and performance for glass data: ")
    print(performance)
    #data_processing(train_list, test_list, glass_mlp, learning_rate, iterations)

def mlp_cancer_data(cancer_mlp, learning_rate, iterations):
    cancer_labels = ["Sample code number", "Uniformity of Clump Thickness", "Uniformity of Cell Size", "Cell Shape", "Marginal Adhesion", "Single Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses", "class"]
    cancer_df = UTILS.import_data(UTILS,"breast-cancer-wisconsin-cleaned.txt", cancer_labels)
    cancer_df.drop(columns=cancer_df.columns[0], axis=1, inplace=True)
    print()
    print("cancer dataframe: ")
    print(cancer_df)

    new_cancer_df = UTILS.min_max_normalization(UTILS, cancer_df)
    print()
    print("normalized cancer dataframe: ")
    print(new_cancer_df)

    cancer_training1,cancer_testing1,cancer_training2,cancer_testing2,cancer_training3,cancer_testing3,cancer_training4,cancer_testing4,cancer_training5,cancer_testing5,cancer_training6,cancer_testing6,cancer_training7,cancer_testing7,cancer_training8,cancer_testing8,cancer_training9,cancer_testing9,cancer_training10,cancer_testing10,cancer_tuning = UTILS.stratify_and_fold_classification(UTILS, new_cancer_df)
    train_list = [cancer_training1, cancer_training2, cancer_training3, cancer_training4, cancer_training5, cancer_training6, cancer_training7, cancer_training8, cancer_training9, cancer_training10]
    test_list = [cancer_testing1, cancer_testing2, cancer_testing3, cancer_testing4, cancer_testing5, cancer_testing6, cancer_testing7, cancer_testing8, cancer_testing9, cancer_testing10]
    classes = [2, 4]
    performance = data_processing(train_list, test_list, cancer_mlp, learning_rate, iterations, classes)
    print()
    print("Final resuld and performance for cancer data: ")
    print(performance)

def mlp_soybean_data(soybean_mlp, learning_rate, iterations):
    soy_labels = ["date","plant-stand","precip","temp","hail","crop-hist","area-damaged","severity","seed-tmt","germination","leaves","lodging","stem-cankers","canker-lesion","fruiting-bodies","external-decay","mycelium","int-discolor","sclerotia","fruit-pods","roots","class"]
    soy_df = UTILS.import_data(UTILS,"soybean-small-cleaned.csv", soy_labels)
    print()
    print("soybean dataframe: ")
    print(soy_df)

    altered_soy_df = UTILS.one_hot_code(UTILS, soy_df, "class")
    print(altered_soy_df)

    new_soybean_df = UTILS.min_max_normalization(UTILS, soy_df)
    print()
    print("normalized glass dataframe: ")
    print(new_soybean_df)

    soy_training1,soy_testing1,soy_training2,soy_testing2,soy_training3,soy_testing3,soy_training4,soy_testing4,soy_training5,soy_testing5,soy_training6,soy_testing6,soy_training7,soy_testing7,soy_training8,soy_testing8,soy_training9,soy_testing9,soy_training10,soy_testing10,soy_tuning = UTILS.stratify_and_fold_classification(UTILS, soy_df)
    train_list = [soy_training1, soy_training2, soy_training3, soy_training4, soy_training5, soy_training6, soy_training7, soy_training8, soy_training9, soy_training10]
    test_list = [soy_testing1, soy_testing2, soy_testing3, soy_testing4, soy_testing5, soy_testing6, soy_testing7, soy_testing8, soy_testing9, soy_testing10]
    classes = [1, 2, 3, 4]
    performance = data_processing(train_list, test_list, soybean_mlp, learning_rate, iterations, classes)
    print()
    print("Final resuld and performance for soybean data: ")
    print(performance)


def data_processing(train_list, test_list, glass_mlp, learing_rate, iterations, classes):
    performance = []
    for i in range(len(train_list)):
        for j in range(len(test_list)):
            glass_training_np = train_list[i].to_numpy()
            print()
            print("glass_training" ,i+1, "numpy: ")
            print(glass_training_np)

            glass_training_targets_df = train_list[i]["class"]
            glass_training_targets_np = glass_training_targets_df.to_numpy()
            print()
            print("glass_training", i+ 1, "_targets")
            print(glass_training_targets_np)

            print("Taining~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            glass_mlp.train(glass_training_np, glass_training_targets_np, iterations, learing_rate)
            #########################################################################
            glass_testing_np = test_list[j].to_numpy()
            print()
            print("glass_testing", j+1, "_numpy: ")
            print(glass_testing_np)
            # Train on our testsets
            print("Testing~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            glass_test_output = glass_mlp.forward_propagate(glass_testing_np)
            print("glass_test_output")
            print(glass_test_output)

            result = glass_mlp._find_max_value(glass_test_output, classes)
            glass_testing_targets = test_list[j]["class"].to_numpy()
            print()
            print("target_output: {}, actual_output: {}, test_set_number: {}".format(glass_testing_targets, result, j+1))
            performance_df = pd.DataFrame(test_list[j]["class"])
            performance_df["predication"] = result
            print(performance_df)
            performance.append(performance_df)
    performance = pd.concat(performance)
    return performance



if __name__ == "__main__":
    learning_rate = 0.1
    iterations = 10
    ###############glass dataset ############################

    # glass_mlp = MLP(10, [6,6], 7)
    # print()
    # print("---------------------------------------------------------------------------------")
    # mlp_glass_data(glass_mlp, learning_rate, iterations)

    ################cancer dataset #######################
    # cancer_mlp = MLP(10, [6,6], 2)
    # print()
    # print("---------------------------------------------------------------------------------")
    # mlp_cancer_data(cancer_mlp, learning_rate, iterations)

    #################soybean dataset ######################
    soybean_mlp = MLP(22, [6,6], 4)
    mlp_soybean_data(soybean_mlp,learning_rate, iterations)

       
    