from mlp import MLP
from utils import UTILS
import pandas as pd
from random import random
import json

def mlp_glass_data(glass_mlp, learning_rate, iterations):
    print(iterations)
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
    loss_dict = get_loss(performance, classes)
    with open('glass_result.txt', 'w+') as convert_file:
     convert_file.write(json.dumps(loss_dict))
    

def mlp_cancer_data(cancer_mlp, learning_rate, iterations):
    cancer_labels = ["Sample code number", "Uniformity of Clump Thickness", "Uniformity of Cell Size", "Cell Shape", "Marginal Adhesion", "Single Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses", "class"]
    cancer_df = UTILS.import_data(UTILS,"breast-cancer-wisconsin-cleaned.txt", cancer_labels)
    cancer_df.drop(columns=cancer_df.columns[0], axis=1, inplace=True)
    # print()
    # print("cancer dataframe: ")
    # print(cancer_df)

    new_cancer_df = UTILS.min_max_normalization(UTILS, cancer_df)
    # print()
    # print("normalized cancer dataframe: ")
    # print(new_cancer_df)

    cancer_training1,cancer_testing1,cancer_training2,cancer_testing2,cancer_training3,cancer_testing3,cancer_training4,cancer_testing4,cancer_training5,cancer_testing5,cancer_training6,cancer_testing6,cancer_training7,cancer_testing7,cancer_training8,cancer_testing8,cancer_training9,cancer_testing9,cancer_training10,cancer_testing10,cancer_tuning = UTILS.stratify_and_fold_classification(UTILS, new_cancer_df)
    train_list = [cancer_training1, cancer_training2, cancer_training3, cancer_training4, cancer_training5, cancer_training6, cancer_training7, cancer_training8, cancer_training9, cancer_training10]
    test_list = [cancer_testing1, cancer_testing2, cancer_testing3, cancer_testing4, cancer_testing5, cancer_testing6, cancer_testing7, cancer_testing8, cancer_testing9, cancer_testing10]
    classes = [2, 4]
    performance = data_processing(train_list, test_list, cancer_mlp, learning_rate, iterations, classes)
    print()
    print("Final resuld and performance for cancer data: ")
    #print(performance)
    loss_dict = get_loss(performance, classes)
    with open('cancer_result.txt', 'w+') as convert_file:
        convert_file.write(json.dumps(loss_dict))

def mlp_soybean_data(soybean_mlp, learning_rate, iterations):
    soy_labels = ["date","plant-stand","precip","temp","hail","crop-hist","area-damaged","severity","seed-tmt","germination","leaves","lodging","stem-cankers","canker-lesion","fruiting-bodies","external-decay","mycelium","int-discolor","sclerotia","fruit-pods","roots","class"]
    soy_df = UTILS.import_data(UTILS,"soybean-small-cleaned.csv", soy_labels)
    # print()
    # print("soybean dataframe: ")
    # print(soy_df)

    classes = soy_df["class"]
    UTILS.one_hot_code(UTILS, classes)

    new_soybean_df = UTILS.min_max_normalization(UTILS, soy_df)
    # print()
    # print("normalized soybean dataframe: ")
    # print(new_soybean_df)

    soy_training1,soy_testing1,soy_training2,soy_testing2,soy_training3,soy_testing3,soy_training4,soy_testing4,soy_training5,soy_testing5,soy_training6,soy_testing6,soy_training7,soy_testing7,soy_training8,soy_testing8,soy_training9,soy_testing9,soy_training10,soy_testing10,soy_tuning = UTILS.stratify_and_fold_classification(UTILS, new_soybean_df)
    train_list = [soy_training1, soy_training2, soy_training3, soy_training4, soy_training5, soy_training6, soy_training7, soy_training8, soy_training9, soy_training10]
    test_list = [soy_testing1, soy_testing2, soy_testing3, soy_testing4, soy_testing5, soy_testing6, soy_testing7, soy_testing8, soy_testing9, soy_testing10]
    classes = [1, 2, 3, 4]
    performance = data_processing(train_list, test_list, soybean_mlp, learning_rate, iterations, classes)
    print()
    # print("Final resuld and performance for soybean data: ")
    # print(performance)
    loss_dict = get_loss(performance, classes)
    with open('glass_result.txt', 'w+') as convert_file:
     convert_file.write(json.dumps(loss_dict))


def data_processing(train_list, test_list, mlp, learing_rate, iterations, classes):
    print(iterations)
    performance = []
    for i in range(len(train_list)):
        training_np = train_list[i].to_numpy()
        # print()
        # print("training" ,i+1, "numpy: ")
        # print(training_np)

        training_targets_df = train_list[i]["class"]
        training_targets_np = training_targets_df.to_numpy()
        # print()
        # print("training", i+ 1, "_targets")
        # print(training_targets_np)

        print("Taining~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        mlp.train_network(training_np, training_targets_np, iterations, learing_rate)
        #########################################################################
        testing_np = test_list[i].to_numpy()
        # print()
        # print("testing", i+1, "_numpy: ")
        # print(testing_np)
        # # Train on our testsets
        print("Testing~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        test_output = mlp.forward_feed(testing_np)
        print("test_output")
        print(test_output)

        result = mlp.find_max_value(test_output, classes)
        testing_targets = test_list[i]["class"].to_numpy()
        print()
        print("target_output: {}, actual_output: {}, test_set_number: {}".format(testing_targets, result, i+1))

        performance_df = pd.DataFrame(test_list[i]["class"])
        performance_df["prediction"] = result
        performance.append(performance_df)
    return performance
    
def get_loss(performances, classes):
    loss_dict = {}
    loss_sum = 0
    couter = 1
    for i in performances:
        loss = UTILS.calculate_loss_function(UTILS, i, classes, "classification")
        loss_sum += loss['F1']
        loss_dict[couter] = loss
        print("test case number: {}, loss: {}".format(couter, loss))
        couter += 1
    print(loss_sum)
    avg_p = loss_sum / couter
    print()
    print("The average F1 score of 10 folds is: ", avg_p)
    return loss_dict 
        

if __name__ == "__main__":
    learning_rate = 0.01
    iterations = 2
    # ###############glass dataset ############################
    # print("ITERATION is: ", iterations)
    # glass_mlp = MLP(10, [5, 5], 7)
    # print()
    # print("---------------------------------------------------------------------------------")
    # mlp_glass_data(glass_mlp, learning_rate, iterations)
    # ################cancer dataset #######################
    # cancer_mlp = MLP(10, [], 2)
    # print()
    # print("---------------------------------------------------------------------------------")
    # mlp_cancer_data(cancer_mlp, learning_rate, iterations)

    # # #################soybean dataset ######################
    # print("---------------------------------------------------------------------------------")
    # soybean_mlp = MLP(22, [12, 12], 4)
    # mlp_soybean_data(soybean_mlp, learning_rate, iterations)
    

       
    