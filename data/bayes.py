import pandas as pd
import numpy as np
import random

from cancer_frequency import cancer_frequency
from glass_frequency import glass_frequency
from votes_frequency import votes_frequency
from iris_frequency import iris_frequency
from soybean_frequency import soy_frequency

# function to bin sets
def bin_set(dataset, labels):
    for col_name, col_data in dataset.iteritems():
            if col_name != "Sample code number" and col_name != "class" and col_name != "id":
                # pd.cut(dataset[col_name], bins=bins, labels=labels, include_lowest=True)
                dataset[col_name] = pd.cut(dataset[col_name], len(labels), labels=labels)
    return dataset

# function to perform likelihood equation
def calculate_likelihood(a_c, n_c, d):
    return (a_c + 1)/(n_c + d)

# function to create likelihood table
def calculate_likelihood_table(train_set, frequency_table, classes):
    features = frequency_table.shape[1]
    likelihood_tables=[]
    segments = int(len(frequency_table)/len(classes))
    for i in range(len(classes)):
        if i == 0:
            class_table = frequency_table.iloc[:segments,:]
            likelihood_table = class_table.apply(calculate_likelihood,args=(train_set[train_set['class'] == classes[i]].shape[0],features))
        elif i == (len(classes)-1):
            class_table = frequency_table.iloc[segments*i:,:]
            likelihood_table = class_table.apply(calculate_likelihood,args=(train_set[train_set['class'] == classes[i]].shape[0],features))
        else:
            class_table = frequency_table.iloc[segments*i:segments*(i+1)]
            likelihood_table = class_table.apply(calculate_likelihood,args=(train_set[train_set['class'] == classes[i]].shape[0],features))
        likelihood_tables.append(likelihood_table)
    combined_likelihood_table = pd.concat(likelihood_tables)
    return combined_likelihood_table

def calculate_feature_product(test_set, train_set_likelihood_table, class_type_dict):
    class_prod_set = {}
    for row_index, row in test_set.iterrows():
        for key in class_type_dict:
            class_prod_set[str(key)] = 1
        for col_idx, value in row.items():
            if col_idx != "Sample code number" and col_idx != "class" and col_idx != "id-num":
                for key in class_type_dict:
                    label = str(value) + "-" + str(key)
                    class_prod_set[str(key)] *= train_set_likelihood_table.loc[label, col_idx]
        for key,value in class_type_dict.items():
            class_prod_set[str(key)] *= value
        classifier = max(class_prod_set, key=class_prod_set.get)
        test_set.at[row_index,'Classifier'] = classifier
    test_set.to_csv('classes.csv')
    return test_set

# function to implement 10 fold and stratify data
def stratify_and_fold(dataset):
    classes = dataset['class'].unique()
    # randomizing the data set
    dataset = dataset.reindex(np.random.permutation(dataset.index)) 
    #reset the index
    dataset = dataset.reset_index(drop=True)
    # split classes
    class_df_set={}
    for c in classes:
        class_df_set[str(c)+'_df'] = dataset[dataset['class'] == c]
    # make 10 different folds of the same size
    class_fold_set = {}
    for key,value in class_df_set.items():
        fold1,fold2,fold3,fold4,fold5,fold6,fold7,fold8,fold9,fold10 = np.array_split(value,10)
        class_fold_set[key+'_folds']=[fold1,fold2,fold3,fold4,fold5,fold6,fold7,fold8,fold9,fold10]
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
    return form_training_test_sets(fold1,fold2,fold3,fold4,fold5,fold6,fold7,fold8,fold9,fold10)

# function to combine folds into training and testing sets
def form_training_test_sets(fold1,fold2,fold3,fold4,fold5,fold6,fold7,fold8,fold9,fold10):
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

# function to create noise data sets from original data sets
def create_noise_set(dataframe, attribute_list):
    # select 10% of features at random
    feature_list = []
    num_of_features = round(len(attribute_list)/10)
    if num_of_features <= 1:
        index = random.randrange(0,len(attribute_list))
        feature_list.append(attribute_list[index])
    else:
        for i in range(num_of_features):
            index = random.randrange(0,len(attribute_list))
            feature_list.append(attribute_list[index])
    for f in feature_list:
        dataframe[f] = np.random.permutation(dataframe[f].values)
    # return the resulting dataframe
    return dataframe

def create_class_type_dict(training_set,classes):
    train_set_N = len(training_set.iloc[: , -1:])
    class_type_dict = {}
    for c in classes:
        class_type_dict[c] = training_set[training_set['class'] == c].shape[0]/train_set_N
    return class_type_dict

#function to create confusion matrix 
def calculate_loss_function(classified_df, class_names):
    confusion_matrix = {}
    actual_class = classified_df["class"].tolist()
    predicted_class = classified_df["Classifier"].tolist()
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

def zero_one_loss_function(classified_df, class_names):
    for row_idx, row in classified_df.iterrows():
        if int(row['class']) == int(row['Classifier']):
            classified_df.at[row_idx,'Classifier'] += 1
            count += 1

def f1_score(classified_df, class_names):
    
    for row_idx, row in classified_df.iterrows():
        if int(row['class']) == int(row['Classifier']):
            classified_df.at[row_idx,'Classifier'] += 1
            count += 1

if __name__ == '__main__':
    print("IMPORTING DATA...")
    # import data into dataframes
    cancer_df = pd.read_csv("breast-cancer-wisconsin-cleaned.txt", sep=",", header=None)
    glass_df = pd.read_csv("glass.data", sep=",", header=None)
    votes_df = pd.read_csv("house-votes-84.data", sep=",", header=None)
    iris_df = pd.read_csv("iris.data", sep=",", header=None)
    soy_df = pd.read_csv("soybean-small-cleaned.csv", sep=",", header=None)

    # label original data frames
    cancer_df.columns = ["Sample code number", "Uniformity of Clump Thickness", "Uniformity of Cell Size", "Cell Shape", "Marginal Adhesion", "Single Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses", "class"]
    glass_df.columns = ['id-num','retractive-index','sodium','magnesium','aluminum','silicon','potasium','calcium','barium','iron','class']
    votes_df.columns = ["class","infants","water","adoption","physician","salvador","religious","satellite","nicaragua","missile","immigration","synfuels","education","superfund","crime","exports","south-africa"]
    iris_df.columns = ['sepal-length','sepal-width','petal-length','petal-width','class']
    soy_df.columns = ["date","plant-stand","precip","temp","hail","crop-hist","area-damaged","severity","seed-tmt","germination","leaves","lodging","stem-cankers","canker-lesion","fruiting-bodies","external-decay","mycelium","int-discolor","sclerotia","fruit-pods","roots","class"]

    print("BINNING DATA...")
    # Binning --> Change from continuous to discrete values
    ########################
    glass_df = bin_set(glass_df, [1,2,3,4,5,6,7,8,9,10,11,12])
    iris_df = bin_set(iris_df, [1, 2, 3, 4, 5])
    #######################

    print("CREATING NOISE DATA SETS...")
    # Create Noise Data Sets
    cancer_noise_df = create_noise_set(cancer_df,["Uniformity of Clump Thickness", "Uniformity of Cell Size", "Cell Shape", "Marginal Adhesion", "Single Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses"])
    glass_noise_df=create_noise_set(glass_df,['retractive-index','sodium','magnesium','aluminum','silicon','potasium','calcium','barium','iron'])
    votes_noise_df = create_noise_set(votes_df,["infants","water","adoption","physician","salvador","religious","satellite","nicaragua","missile","immigration","synfuels","education","superfund","crime","exports","south-africa"])
    iris_noise_df=create_noise_set(iris_df,['sepal-length','sepal-width','petal-length','petal-width'])
    soy_noise_df = create_noise_set(soy_df,["date","plant-stand","precip","temp","hail","crop-hist","area-damaged","severity","seed-tmt","germination","leaves","lodging","stem-cankers","canker-lesion","fruiting-bodies","external-decay","mycelium","int-discolor","sclerotia","fruit-pods","roots"])

    # Calculate variables for later reference
    cancer_classes = cancer_df['class'].unique()
    glass_classes = glass_df['class'].unique()
    votes_classes = votes_df['class'].unique()
    iris_classes = iris_df['class'].unique()
    soy_classes = soy_df['class'].unique()

    print("STRATIFING DATA AND CREATING FOLDS...")
    # Create Original training and testing dataframes
    cancer_training1,cancer_testing1,cancer_training2,cancer_testing2,cancer_training3,cancer_testing3,cancer_training4,cancer_testing4,cancer_training5,cancer_testing5,cancer_training6,cancer_testing6,cancer_training7,cancer_testing7,cancer_training8,cancer_testing8,cancer_training9,cancer_testing9,cancer_training10,cancer_testing10 = stratify_and_fold(cancer_df)
    glass_training1,glass_testing1,glass_training2,glass_testing2,glass_training3,glass_testing3,glass_training4,glass_testing4,glass_training5,glass_testing5,glass_training6,glass_testing6,glass_training7,glass_testing7,glass_training8,glass_testing8,glass_training9,glass_testing9,glass_training10,glass_testing10 = stratify_and_fold(glass_df)
    votes_training1,votes_testing1,votes_training2,votes_testing2,votes_training3,votes_testing3,votes_training4,votes_testing4,votes_training5,votes_testing5,votes_training6,votes_testing6,votes_training7,votes_testing7,votes_training8,votes_testing8,votes_training9,votes_testing9,votes_training10,votes_testing10 = stratify_and_fold(votes_df)
    iris_training1,iris_testing1,iris_training2,iris_testing2,iris_training3,iris_testing3,iris_training4,iris_testing4,iris_training5,iris_testing5,iris_training6,iris_testing6,iris_training7,iris_testing7,iris_training8,iris_testing8,iris_training9,iris_testing9,iris_training10,iris_testing10 = stratify_and_fold(iris_df)
    soy_training1,soy_testing1,soy_training2,soy_testing2,soy_training3,soy_testing3,soy_training4,soy_testing4,soy_training5,soy_testing5,soy_training6,soy_testing6,soy_training7,soy_testing7,soy_training8,soy_testing8,soy_training9,soy_testing9,soy_training10,soy_testing10 = stratify_and_fold(soy_df)

    # Create Noise training and testing dataframes
    cancer_noise_training1,cancer_noise_testing1,cancer_noise_training2,cancer_noise_testing2,cancer_noise_training3,cancer_noise_testing3,cancer_noise_training4,cancer_noise_testing4,cancer_noise_training5,cancer_noise_testing5,cancer_noise_training6,cancer_noise_testing6,cancer_noise_training7,cancer_noise_testing7,cancer_noise_training8,cancer_noise_testing8,cancer_noise_training9,cancer_noise_testing9,cancer_noise_training10,cancer_noise_testing10 = stratify_and_fold(cancer_noise_df)
    glass_noise_training1,glass_noise_testing1,glass_noise_training2,glass_noise_testing2,glass_noise_training3,glass_noise_testing3,glass_noise_training4,glass_noise_testing4,glass_noise_training5,glass_noise_testing5,glass_noise_training6,glass_noise_testing6,glass_noise_training7,glass_noise_testing7,glass_noise_training8,glass_noise_testing8,glass_noise_training9,glass_noise_testing9,glass_noise_training10,glass_noise_testing10 = stratify_and_fold(glass_noise_df)
    votes_noise_training1,votes_noise_testing1,votes_noise_training2,votes_noise_testing2,votes_noise_training3,votes_noise_testing3,votes_noise_training4,votes_noise_testing4,votes_noise_training5,votes_noise_testing5,votes_noise_training6,votes_noise_testing6,votes_noise_training7,votes_noise_testing7,votes_noise_training8,votes_noise_testing8,votes_noise_training9,votes_noise_testing9,votes_noise_training10,votes_noise_testing10 = stratify_and_fold(votes_noise_df)
    iris_noise_training1,iris_noise_testing1,iris_noise_training2,iris_noise_testing2,iris_noise_training3,iris_noise_testing3,iris_noise_training4,iris_noise_testing4,iris_noise_training5,iris_noise_testing5,iris_noise_training6,iris_noise_testing6,iris_noise_training7,iris_noise_testing7,iris_noise_training8,iris_noise_testing8,iris_noise_training9,iris_noise_testing9,iris_noise_training10,iris_noise_testing10 = stratify_and_fold(iris_noise_df)
    soy_noise_training1,soy_noise_testing1,soy_noise_training2,soy_noise_testing2,soy_noise_training3,soy_noise_testing3,soy_noise_training4,soy_noise_testing4,soy_noise_training5,soy_noise_testing5,soy_noise_training6,soy_noise_testing6,soy_noise_training7,soy_noise_testing7,soy_noise_training8,soy_noise_testing8,soy_noise_training9,soy_noise_testing9,soy_noise_training10,soy_noise_testing10 = stratify_and_fold(soy_noise_df)

    print("CREATING FREQUENCY TABLES...")
    # Calculate Original frequency tables
    cancer_train1_frequency_table = cancer_frequency(cancer_training1)
    cancer_train2_frequency_table = cancer_frequency(cancer_training2)
    cancer_train3_frequency_table = cancer_frequency(cancer_training3)
    cancer_train4_frequency_table = cancer_frequency(cancer_training4)
    cancer_train5_frequency_table = cancer_frequency(cancer_training5)
    cancer_train6_frequency_table = cancer_frequency(cancer_training6)
    cancer_train7_frequency_table = cancer_frequency(cancer_training7)
    cancer_train8_frequency_table = cancer_frequency(cancer_training8)
    cancer_train9_frequency_table = cancer_frequency(cancer_training9)
    cancer_train10_frequency_table = cancer_frequency(cancer_training10)

    glass_train1_frequency_table = glass_frequency(glass_training1)
    glass_train2_frequency_table = glass_frequency(glass_training2)
    glass_train3_frequency_table = glass_frequency(glass_training3)
    glass_train4_frequency_table = glass_frequency(glass_training4)
    glass_train5_frequency_table = glass_frequency(glass_training5)
    glass_train6_frequency_table = glass_frequency(glass_training6)
    glass_train7_frequency_table = glass_frequency(glass_training7)
    glass_train8_frequency_table = glass_frequency(glass_training8)
    glass_train9_frequency_table = glass_frequency(glass_training9)
    glass_train10_frequency_table = glass_frequency(glass_training10)
    
    votes_train1_frequency_table = votes_frequency(votes_training1)
    votes_train2_frequency_table = votes_frequency(votes_training2)
    votes_train3_frequency_table = votes_frequency(votes_training3)
    votes_train4_frequency_table = votes_frequency(votes_training4)
    votes_train5_frequency_table = votes_frequency(votes_training5)
    votes_train6_frequency_table = votes_frequency(votes_training6)
    votes_train7_frequency_table = votes_frequency(votes_training7)
    votes_train8_frequency_table = votes_frequency(votes_training8)
    votes_train9_frequency_table = votes_frequency(votes_training9)
    votes_train10_frequency_table = votes_frequency(votes_training10)
    
    iris_train1_frequency_table = iris_frequency(iris_training1)
    iris_train2_frequency_table = iris_frequency(iris_training2)
    iris_train3_frequency_table = iris_frequency(iris_training3)
    iris_train4_frequency_table = iris_frequency(iris_training4)
    iris_train5_frequency_table = iris_frequency(iris_training5)
    iris_train6_frequency_table = iris_frequency(iris_training6)
    iris_train7_frequency_table = iris_frequency(iris_training7)
    iris_train8_frequency_table = iris_frequency(iris_training8)
    iris_train9_frequency_table = iris_frequency(iris_training9)
    iris_train10_frequency_table = iris_frequency(iris_training10)

    soy_train1_frequency_table = soy_frequency(soy_training1)
    soy_train2_frequency_table = soy_frequency(soy_training2)
    soy_train3_frequency_table = soy_frequency(soy_training3)
    soy_train4_frequency_table = soy_frequency(soy_training4)
    soy_train5_frequency_table = soy_frequency(soy_training5)
    soy_train6_frequency_table = soy_frequency(soy_training6)
    soy_train7_frequency_table = soy_frequency(soy_training7)
    soy_train8_frequency_table = soy_frequency(soy_training8)
    soy_train9_frequency_table = soy_frequency(soy_training9)
    soy_train10_frequency_table = soy_frequency(soy_training10)

    # Calculate Noise frequency tables
    cancer_noise_train1_frequency_table = cancer_frequency(cancer_noise_training1)
    cancer_noise_train2_frequency_table = cancer_frequency(cancer_noise_training2)
    cancer_noise_train3_frequency_table = cancer_frequency(cancer_noise_training3)
    cancer_noise_train4_frequency_table = cancer_frequency(cancer_noise_training4)
    cancer_noise_train5_frequency_table = cancer_frequency(cancer_noise_training5)
    cancer_noise_train6_frequency_table = cancer_frequency(cancer_noise_training6)
    cancer_noise_train7_frequency_table = cancer_frequency(cancer_noise_training7)
    cancer_noise_train8_frequency_table = cancer_frequency(cancer_noise_training8)
    cancer_noise_train9_frequency_table = cancer_frequency(cancer_noise_training9)
    cancer_noise_train10_frequency_table = cancer_frequency(cancer_noise_training10)
    
    glass_noise_train1_frequency_table = glass_frequency(glass_noise_training1)
    glass_noise_train2_frequency_table = glass_frequency(glass_noise_training2)
    glass_noise_train3_frequency_table = glass_frequency(glass_noise_training3)
    glass_noise_train4_frequency_table = glass_frequency(glass_noise_training4)
    glass_noise_train5_frequency_table = glass_frequency(glass_noise_training5)
    glass_noise_train6_frequency_table = glass_frequency(glass_noise_training6)
    glass_noise_train7_frequency_table = glass_frequency(glass_noise_training7)
    glass_noise_train8_frequency_table = glass_frequency(glass_noise_training8)
    glass_noise_train9_frequency_table = glass_frequency(glass_noise_training9)
    glass_noise_train10_frequency_table = glass_frequency(glass_noise_training10)
    
    votes_noise_train1_frequency_table = votes_frequency(votes_noise_training1)
    votes_noise_train2_frequency_table = votes_frequency(votes_noise_training2)
    votes_noise_train3_frequency_table = votes_frequency(votes_noise_training3)
    votes_noise_train4_frequency_table = votes_frequency(votes_noise_training4)
    votes_noise_train5_frequency_table = votes_frequency(votes_noise_training5)
    votes_noise_train6_frequency_table = votes_frequency(votes_noise_training6)
    votes_noise_train7_frequency_table = votes_frequency(votes_noise_training7)
    votes_noise_train8_frequency_table = votes_frequency(votes_noise_training8)
    votes_noise_train9_frequency_table = votes_frequency(votes_noise_training9)
    votes_noise_train10_frequency_table = votes_frequency(votes_noise_training10)
    
    iris_noise_train1_frequency_table = iris_frequency(iris_noise_training1)
    iris_noise_train2_frequency_table = iris_frequency(iris_noise_training2)
    iris_noise_train3_frequency_table = iris_frequency(iris_noise_training3)
    iris_noise_train4_frequency_table = iris_frequency(iris_noise_training4)
    iris_noise_train5_frequency_table = iris_frequency(iris_noise_training5)
    iris_noise_train6_frequency_table = iris_frequency(iris_noise_training6)
    iris_noise_train7_frequency_table = iris_frequency(iris_noise_training7)
    iris_noise_train8_frequency_table = iris_frequency(iris_noise_training8)
    iris_noise_train9_frequency_table = iris_frequency(iris_noise_training9)
    iris_noise_train10_frequency_table = iris_frequency(iris_noise_training10)

    soy_noise_train1_frequency_table = soy_frequency(soy_noise_training1)
    soy_noise_train2_frequency_table = soy_frequency(soy_noise_training2)
    soy_noise_train3_frequency_table = soy_frequency(soy_noise_training3)
    soy_noise_train4_frequency_table = soy_frequency(soy_noise_training4)
    soy_noise_train5_frequency_table = soy_frequency(soy_noise_training5)
    soy_noise_train6_frequency_table = soy_frequency(soy_noise_training6)
    soy_noise_train7_frequency_table = soy_frequency(soy_noise_training7)
    soy_noise_train8_frequency_table = soy_frequency(soy_noise_training8)
    soy_noise_train9_frequency_table = soy_frequency(soy_noise_training9)
    soy_noise_train10_frequency_table = soy_frequency(soy_noise_training10)

    print("CREATING CLASS CONDITIONAL PROBAILITY TABLES...")
    # Calculate Original Likelihood Tables
    cancer_train1_likehood_table = calculate_likelihood_table(cancer_training1, cancer_train1_frequency_table,cancer_classes)
    cancer_train2_likehood_table = calculate_likelihood_table(cancer_training2, cancer_train2_frequency_table,cancer_classes)
    cancer_train3_likehood_table = calculate_likelihood_table(cancer_training3, cancer_train3_frequency_table,cancer_classes)
    cancer_train4_likehood_table = calculate_likelihood_table(cancer_training4, cancer_train4_frequency_table,cancer_classes)
    cancer_train5_likehood_table = calculate_likelihood_table(cancer_training5, cancer_train5_frequency_table,cancer_classes)
    cancer_train6_likehood_table = calculate_likelihood_table(cancer_training6, cancer_train6_frequency_table,cancer_classes)
    cancer_train7_likehood_table = calculate_likelihood_table(cancer_training7, cancer_train7_frequency_table,cancer_classes)
    cancer_train8_likehood_table = calculate_likelihood_table(cancer_training8, cancer_train8_frequency_table,cancer_classes)
    cancer_train9_likehood_table = calculate_likelihood_table(cancer_training9, cancer_train9_frequency_table,cancer_classes)
    cancer_train10_likehood_table = calculate_likelihood_table(cancer_training10, cancer_train10_frequency_table,cancer_classes)
    
    glass_train1_likehood_table = calculate_likelihood_table(glass_training1, glass_train1_frequency_table,glass_classes)
    glass_train2_likehood_table = calculate_likelihood_table(glass_training2, glass_train2_frequency_table,glass_classes)
    glass_train3_likehood_table = calculate_likelihood_table(glass_training3, glass_train3_frequency_table,glass_classes)
    glass_train4_likehood_table = calculate_likelihood_table(glass_training4, glass_train4_frequency_table,glass_classes)
    glass_train5_likehood_table = calculate_likelihood_table(glass_training5, glass_train5_frequency_table,glass_classes)
    glass_train6_likehood_table = calculate_likelihood_table(glass_training6, glass_train6_frequency_table,glass_classes)
    glass_train7_likehood_table = calculate_likelihood_table(glass_training7, glass_train7_frequency_table,glass_classes)
    glass_train8_likehood_table = calculate_likelihood_table(glass_training8, glass_train8_frequency_table,glass_classes)
    glass_train9_likehood_table = calculate_likelihood_table(glass_training9, glass_train9_frequency_table,glass_classes)
    glass_train10_likehood_table = calculate_likelihood_table(glass_training10, glass_train10_frequency_table,glass_classes)
    
    votes_train1_likehood_table = calculate_likelihood_table(votes_training1, votes_train1_frequency_table,votes_classes)
    votes_train2_likehood_table = calculate_likelihood_table(votes_training2, votes_train2_frequency_table,votes_classes)
    votes_train3_likehood_table = calculate_likelihood_table(votes_training3, votes_train3_frequency_table,votes_classes)
    votes_train4_likehood_table = calculate_likelihood_table(votes_training4, votes_train4_frequency_table,votes_classes)
    votes_train5_likehood_table = calculate_likelihood_table(votes_training5, votes_train5_frequency_table,votes_classes)
    votes_train6_likehood_table = calculate_likelihood_table(votes_training6, votes_train6_frequency_table,votes_classes)
    votes_train7_likehood_table = calculate_likelihood_table(votes_training7, votes_train7_frequency_table,votes_classes)
    votes_train8_likehood_table = calculate_likelihood_table(votes_training8, votes_train8_frequency_table,votes_classes)
    votes_train9_likehood_table = calculate_likelihood_table(votes_training9, votes_train9_frequency_table,votes_classes)
    votes_train10_likehood_table = calculate_likelihood_table(votes_training10, votes_train10_frequency_table,votes_classes)
    
    iris_train1_likehood_table = calculate_likelihood_table(iris_training1, iris_train1_frequency_table,iris_classes)
    iris_train2_likehood_table = calculate_likelihood_table(iris_training2, iris_train2_frequency_table,iris_classes)
    iris_train3_likehood_table = calculate_likelihood_table(iris_training3, iris_train3_frequency_table,iris_classes)
    iris_train4_likehood_table = calculate_likelihood_table(iris_training4, iris_train4_frequency_table,iris_classes)
    iris_train5_likehood_table = calculate_likelihood_table(iris_training5, iris_train5_frequency_table,iris_classes)
    iris_train6_likehood_table = calculate_likelihood_table(iris_training6, iris_train6_frequency_table,iris_classes)
    iris_train7_likehood_table = calculate_likelihood_table(iris_training7, iris_train7_frequency_table,iris_classes)
    iris_train8_likehood_table = calculate_likelihood_table(iris_training8, iris_train8_frequency_table,iris_classes)
    iris_train9_likehood_table = calculate_likelihood_table(iris_training9, iris_train9_frequency_table,iris_classes)
    iris_train10_likehood_table = calculate_likelihood_table(iris_training10, iris_train10_frequency_table,iris_classes)
    
    soy_train1_likehood_table = calculate_likelihood_table(soy_training1, soy_train1_frequency_table,soy_classes)
    soy_train2_likehood_table = calculate_likelihood_table(soy_training2, soy_train2_frequency_table,soy_classes)
    soy_train3_likehood_table = calculate_likelihood_table(soy_training3, soy_train3_frequency_table,soy_classes)
    soy_train4_likehood_table = calculate_likelihood_table(soy_training4, soy_train4_frequency_table,soy_classes)
    soy_train5_likehood_table = calculate_likelihood_table(soy_training5, soy_train5_frequency_table,soy_classes)
    soy_train6_likehood_table = calculate_likelihood_table(soy_training6, soy_train6_frequency_table,soy_classes)
    soy_train7_likehood_table = calculate_likelihood_table(soy_training7, soy_train7_frequency_table,soy_classes)
    soy_train8_likehood_table = calculate_likelihood_table(soy_training8, soy_train8_frequency_table,soy_classes)
    soy_train9_likehood_table = calculate_likelihood_table(soy_training9, soy_train9_frequency_table,soy_classes)
    soy_train10_likehood_table = calculate_likelihood_table(soy_training10, soy_train10_frequency_table,soy_classes)

    # Calculate Noise Likelihood Tables
    cancer_noise_train1_likehood_table = calculate_likelihood_table(cancer_noise_training1, cancer_noise_train1_frequency_table,cancer_classes)
    cancer_noise_train2_likehood_table = calculate_likelihood_table(cancer_noise_training2, cancer_noise_train2_frequency_table,cancer_classes)
    cancer_noise_train3_likehood_table = calculate_likelihood_table(cancer_noise_training3, cancer_noise_train3_frequency_table,cancer_classes)
    cancer_noise_train4_likehood_table = calculate_likelihood_table(cancer_noise_training4, cancer_noise_train4_frequency_table,cancer_classes)
    cancer_noise_train5_likehood_table = calculate_likelihood_table(cancer_noise_training5, cancer_noise_train5_frequency_table,cancer_classes)
    cancer_noise_train6_likehood_table = calculate_likelihood_table(cancer_noise_training6, cancer_noise_train6_frequency_table,cancer_classes)
    cancer_noise_train7_likehood_table = calculate_likelihood_table(cancer_noise_training7, cancer_noise_train7_frequency_table,cancer_classes)
    cancer_noise_train8_likehood_table = calculate_likelihood_table(cancer_noise_training8, cancer_noise_train8_frequency_table,cancer_classes)
    cancer_noise_train9_likehood_table = calculate_likelihood_table(cancer_noise_training9, cancer_noise_train9_frequency_table,cancer_classes)
    cancer_noise_train10_likehood_table = calculate_likelihood_table(cancer_noise_training10, cancer_noise_train10_frequency_table,cancer_classes)
    
    glass_noise_train1_likehood_table = calculate_likelihood_table(glass_noise_training1, glass_noise_train1_frequency_table,glass_classes)
    glass_noise_train2_likehood_table = calculate_likelihood_table(glass_noise_training2, glass_noise_train2_frequency_table,glass_classes)
    glass_noise_train3_likehood_table = calculate_likelihood_table(glass_noise_training3, glass_noise_train3_frequency_table,glass_classes)
    glass_noise_train4_likehood_table = calculate_likelihood_table(glass_noise_training4, glass_noise_train4_frequency_table,glass_classes)
    glass_noise_train5_likehood_table = calculate_likelihood_table(glass_noise_training5, glass_noise_train5_frequency_table,glass_classes)
    glass_noise_train6_likehood_table = calculate_likelihood_table(glass_noise_training6, glass_noise_train6_frequency_table,glass_classes)
    glass_noise_train7_likehood_table = calculate_likelihood_table(glass_noise_training7, glass_noise_train7_frequency_table,glass_classes)
    glass_noise_train8_likehood_table = calculate_likelihood_table(glass_noise_training8, glass_noise_train8_frequency_table,glass_classes)
    glass_noise_train9_likehood_table = calculate_likelihood_table(glass_noise_training9, glass_noise_train9_frequency_table,glass_classes)
    glass_noise_train10_likehood_table = calculate_likelihood_table(glass_noise_training10, glass_noise_train10_frequency_table,glass_classes)
    
    votes_noise_train1_likehood_table = calculate_likelihood_table(votes_noise_training1, votes_noise_train1_frequency_table,votes_classes)
    votes_noise_train2_likehood_table = calculate_likelihood_table(votes_noise_training2, votes_noise_train2_frequency_table,votes_classes)
    votes_noise_train3_likehood_table = calculate_likelihood_table(votes_noise_training3, votes_noise_train3_frequency_table,votes_classes)
    votes_noise_train4_likehood_table = calculate_likelihood_table(votes_noise_training4, votes_noise_train4_frequency_table,votes_classes)
    votes_noise_train5_likehood_table = calculate_likelihood_table(votes_noise_training5, votes_noise_train5_frequency_table,votes_classes)
    votes_noise_train6_likehood_table = calculate_likelihood_table(votes_noise_training6, votes_noise_train6_frequency_table,votes_classes)
    votes_noise_train7_likehood_table = calculate_likelihood_table(votes_noise_training7, votes_noise_train7_frequency_table,votes_classes)
    votes_noise_train8_likehood_table = calculate_likelihood_table(votes_noise_training8, votes_noise_train8_frequency_table,votes_classes)
    votes_noise_train9_likehood_table = calculate_likelihood_table(votes_noise_training9, votes_noise_train9_frequency_table,votes_classes)
    votes_noise_train10_likehood_table = calculate_likelihood_table(votes_noise_training10, votes_noise_train10_frequency_table,votes_classes)
    
    iris_noise_train1_likehood_table = calculate_likelihood_table(iris_noise_training1, iris_noise_train1_frequency_table,iris_classes)
    iris_noise_train2_likehood_table = calculate_likelihood_table(iris_noise_training2, iris_noise_train2_frequency_table,iris_classes)
    iris_noise_train3_likehood_table = calculate_likelihood_table(iris_noise_training3, iris_noise_train3_frequency_table,iris_classes)
    iris_noise_train4_likehood_table = calculate_likelihood_table(iris_noise_training4, iris_noise_train4_frequency_table,iris_classes)
    iris_noise_train5_likehood_table = calculate_likelihood_table(iris_noise_training5, iris_noise_train5_frequency_table,iris_classes)
    iris_noise_train6_likehood_table = calculate_likelihood_table(iris_noise_training6, iris_noise_train6_frequency_table,iris_classes)
    iris_noise_train7_likehood_table = calculate_likelihood_table(iris_noise_training7, iris_noise_train7_frequency_table,iris_classes)
    iris_noise_train8_likehood_table = calculate_likelihood_table(iris_noise_training8, iris_noise_train8_frequency_table,iris_classes)
    iris_noise_train9_likehood_table = calculate_likelihood_table(iris_noise_training9, iris_noise_train9_frequency_table,iris_classes)
    iris_noise_train10_likehood_table = calculate_likelihood_table(iris_noise_training10, iris_noise_train10_frequency_table,iris_classes)
    
    soy_noise_train1_likehood_table = calculate_likelihood_table(soy_noise_training1, soy_noise_train1_frequency_table,soy_classes)
    soy_noise_train2_likehood_table = calculate_likelihood_table(soy_noise_training2, soy_noise_train2_frequency_table,soy_classes)
    soy_noise_train3_likehood_table = calculate_likelihood_table(soy_noise_training3, soy_noise_train3_frequency_table,soy_classes)
    soy_noise_train4_likehood_table = calculate_likelihood_table(soy_noise_training4, soy_noise_train4_frequency_table,soy_classes)
    soy_noise_train5_likehood_table = calculate_likelihood_table(soy_noise_training5, soy_noise_train5_frequency_table,soy_classes)
    soy_noise_train6_likehood_table = calculate_likelihood_table(soy_noise_training6, soy_noise_train6_frequency_table,soy_classes)
    soy_noise_train7_likehood_table = calculate_likelihood_table(soy_noise_training7, soy_noise_train7_frequency_table,soy_classes)
    soy_noise_train8_likehood_table = calculate_likelihood_table(soy_noise_training8, soy_noise_train8_frequency_table,soy_classes)
    soy_noise_train9_likehood_table = calculate_likelihood_table(soy_noise_training9, soy_noise_train9_frequency_table,soy_classes)
    soy_noise_train10_likehood_table = calculate_likelihood_table(soy_noise_training10, soy_noise_train10_frequency_table,soy_classes)

    print("PERFORMING CLASSIFICATION ON TEST SETS...")
    # Perform classification on original test sets
    cancer_train1_class_type_dict = create_class_type_dict(cancer_training1,cancer_classes)
    cancer_test1_results = calculate_feature_product(cancer_testing1, cancer_train1_likehood_table, cancer_train1_class_type_dict)
    # print("--------------------------------")
    # print("RESULTS")
    # print(cancer_test1_results)
    cancer_train2_class_type_dict = create_class_type_dict(cancer_training2,cancer_classes)
    cancer_test2_results = calculate_feature_product(cancer_testing2, cancer_train2_likehood_table, cancer_train2_class_type_dict)
    cancer_train3_class_type_dict = create_class_type_dict(cancer_training3,cancer_classes)
    cancer_test3_results = calculate_feature_product(cancer_testing3, cancer_train3_likehood_table, cancer_train3_class_type_dict)
    cancer_train4_class_type_dict = create_class_type_dict(cancer_training4,cancer_classes)
    cancer_test4_results = calculate_feature_product(cancer_testing4, cancer_train4_likehood_table, cancer_train4_class_type_dict)
    cancer_train5_class_type_dict = create_class_type_dict(cancer_training5,cancer_classes)
    cancer_test5_results = calculate_feature_product(cancer_testing5, cancer_train5_likehood_table, cancer_train5_class_type_dict)
    cancer_train6_class_type_dict = create_class_type_dict(cancer_training6,cancer_classes)
    cancer_test6_results = calculate_feature_product(cancer_testing6, cancer_train6_likehood_table, cancer_train6_class_type_dict)
    cancer_train7_class_type_dict = create_class_type_dict(cancer_training7,cancer_classes)
    cancer_test7_results = calculate_feature_product(cancer_testing7, cancer_train7_likehood_table, cancer_train7_class_type_dict)
    cancer_train8_class_type_dict = create_class_type_dict(cancer_training8,cancer_classes)
    cancer_test8_results = calculate_feature_product(cancer_testing8, cancer_train8_likehood_table, cancer_train8_class_type_dict)
    cancer_train9_class_type_dict = create_class_type_dict(cancer_training9,cancer_classes)
    cancer_test9_results = calculate_feature_product(cancer_testing9, cancer_train9_likehood_table, cancer_train9_class_type_dict)
    cancer_train10_class_type_dict = create_class_type_dict(cancer_training10,cancer_classes)
    cancer_test10_results = calculate_feature_product(cancer_testing10, cancer_train10_likehood_table, cancer_train10_class_type_dict)

    glass_train1_class_type_dict = create_class_type_dict(glass_training1,glass_classes)
    glass_test1_results = calculate_feature_product(glass_testing1, glass_train1_likehood_table, glass_train1_class_type_dict)
    print("--------------------------------")
    print("TEST1")
    count = 0
    for row_idx, row in glass_test1_results.iterrows():
        if int(row['class']) == int(row['Classifier']):
            count += 1
    print(count,"out of",glass_test1_results.shape[0])
    glass_train2_class_type_dict = create_class_type_dict(glass_training2,glass_classes)
    glass_test2_results = calculate_feature_product(glass_testing2, glass_train2_likehood_table, glass_train2_class_type_dict)
    print("TEST 2")
    count = 0
    for row_idx, row in glass_test2_results.iterrows():
        if int(row['class']) == int(row['Classifier']):
            count += 1
    print(count,"out of",glass_test2_results.shape[0])
    glass_train3_class_type_dict = create_class_type_dict(glass_training3,glass_classes)
    glass_test3_results = calculate_feature_product(glass_testing3, glass_train3_likehood_table, glass_train3_class_type_dict)
    print("TEST3")
    count = 0
    for row_idx, row in glass_test3_results.iterrows():
        if int(row['class']) == int(row['Classifier']):
            count += 1
    print(count,"out of",glass_test3_results.shape[0])
    glass_train4_class_type_dict = create_class_type_dict(glass_training4,glass_classes)
    glass_test4_results = calculate_feature_product(glass_testing4, glass_train4_likehood_table, glass_train4_class_type_dict)
    print("TEST4")
    count = 0
    for row_idx, row in glass_test4_results.iterrows():
        if int(row['class']) == int(row['Classifier']):
            count += 1
    print(count,"out of",glass_test4_results.shape[0])
    glass_train5_class_type_dict = create_class_type_dict(glass_training5,glass_classes)
    glass_test5_results = calculate_feature_product(glass_testing5, glass_train5_likehood_table, glass_train5_class_type_dict)
    print("TEST5")
    count = 0
    for row_idx, row in glass_test5_results.iterrows():
        if int(row['class']) == int(row['Classifier']):
            count += 1
    print(count,"out of",glass_test5_results.shape[0])
    glass_train6_class_type_dict = create_class_type_dict(glass_training6,glass_classes)
    glass_test6_results = calculate_feature_product(glass_testing6, glass_train6_likehood_table, glass_train6_class_type_dict)
    print("TEST6")
    count = 0
    for row_idx, row in glass_test6_results.iterrows():
        if int(row['class']) == int(row['Classifier']):
            count += 1
    print(count,"out of",glass_test6_results.shape[0])
    glass_train7_class_type_dict = create_class_type_dict(glass_training7,glass_classes)
    glass_test7_results = calculate_feature_product(glass_testing7, glass_train7_likehood_table, glass_train7_class_type_dict)
    print("TEST7")
    count = 0
    for row_idx, row in glass_test7_results.iterrows():
        if int(row['class']) == int(row['Classifier']):
            count += 1
    print(count,"out of",glass_test7_results.shape[0])
    glass_train8_class_type_dict = create_class_type_dict(glass_training8,glass_classes)
    glass_test8_results = calculate_feature_product(glass_testing8, glass_train8_likehood_table, glass_train8_class_type_dict)
    print("TEST8")
    count = 0
    for row_idx, row in glass_test8_results.iterrows():
        if int(row['class']) == int(row['Classifier']):
            count += 1
    print(count,"out of",glass_test8_results.shape[0])
    glass_train9_class_type_dict = create_class_type_dict(glass_training9,glass_classes)
    glass_test9_results = calculate_feature_product(glass_testing9, glass_train9_likehood_table, glass_train9_class_type_dict)
    print("TEST9")
    count = 0
    for row_idx, row in glass_test9_results.iterrows():
        if int(row['class']) == int(row['Classifier']):
            count += 1
    print(count,"out of",glass_test9_results.shape[0])
    glass_train10_class_type_dict = create_class_type_dict(glass_training10,glass_classes)
    glass_test10_results = calculate_feature_product(glass_testing10, glass_train10_likehood_table, glass_train10_class_type_dict)
    print("TEST10")
    count = 0
    for row_idx, row in glass_test10_results.iterrows():
        if int(row['class']) == int(row['Classifier']):
            count += 1
    print(count,"out of",glass_test10_results.shape[0])

    votes_train1_class_type_dict = create_class_type_dict(votes_training1,votes_classes)
    votes_test1_results = calculate_feature_product(votes_testing1, votes_train1_likehood_table, votes_train1_class_type_dict)
    # print("--------------------------------")
    # print("RESULTS")
    # print(votes_test1_results)
    votes_train2_class_type_dict = create_class_type_dict(votes_training2,votes_classes)
    votes_test2_results = calculate_feature_product(votes_testing2, votes_train2_likehood_table, votes_train2_class_type_dict)
    votes_train3_class_type_dict = create_class_type_dict(votes_training3,votes_classes)
    votes_test3_results = calculate_feature_product(votes_testing3, votes_train3_likehood_table, votes_train3_class_type_dict)
    votes_train4_class_type_dict = create_class_type_dict(votes_training4,votes_classes)
    votes_test4_results = calculate_feature_product(votes_testing4, votes_train4_likehood_table, votes_train4_class_type_dict)
    votes_train5_class_type_dict = create_class_type_dict(votes_training5,votes_classes)
    votes_test5_results = calculate_feature_product(votes_testing5, votes_train5_likehood_table, votes_train5_class_type_dict)
    votes_train6_class_type_dict = create_class_type_dict(votes_training6,votes_classes)
    votes_test6_results = calculate_feature_product(votes_testing6, votes_train6_likehood_table, votes_train6_class_type_dict)
    votes_train7_class_type_dict = create_class_type_dict(votes_training7,votes_classes)
    votes_test7_results = calculate_feature_product(votes_testing7, votes_train7_likehood_table, votes_train7_class_type_dict)
    votes_train8_class_type_dict = create_class_type_dict(votes_training8,votes_classes)
    votes_test8_results = calculate_feature_product(votes_testing8, votes_train8_likehood_table, votes_train8_class_type_dict)
    votes_train9_class_type_dict = create_class_type_dict(votes_training9,votes_classes)
    votes_test9_results = calculate_feature_product(votes_testing9, votes_train9_likehood_table, votes_train9_class_type_dict)
    votes_train10_class_type_dict = create_class_type_dict(votes_training10,votes_classes)
    votes_test10_results = calculate_feature_product(votes_testing10, votes_train10_likehood_table, votes_train10_class_type_dict)

    iris_train1_class_type_dict = create_class_type_dict(iris_training1,iris_classes)
    iris_test1_results = calculate_feature_product(iris_testing1, iris_train1_likehood_table, iris_train1_class_type_dict)
    # print("--------------------------------")
    # print("RESULTS")
    # print(iris_test1_results)
    iris_train2_class_type_dict = create_class_type_dict(iris_training2,iris_classes)
    iris_test2_results = calculate_feature_product(iris_testing2, iris_train2_likehood_table, iris_train2_class_type_dict)
    # print(iris_test2_results)
    iris_train3_class_type_dict = create_class_type_dict(iris_training3,iris_classes)
    iris_test3_results = calculate_feature_product(iris_testing3, iris_train3_likehood_table, iris_train3_class_type_dict)
    # print(iris_test3_results)
    iris_train4_class_type_dict = create_class_type_dict(iris_training4,iris_classes)
    iris_test4_results = calculate_feature_product(iris_testing4, iris_train4_likehood_table, iris_train4_class_type_dict)
    # print(iris_test4_results)
    iris_train5_class_type_dict = create_class_type_dict(iris_training5,iris_classes)
    iris_test5_results = calculate_feature_product(iris_testing5, iris_train5_likehood_table, iris_train5_class_type_dict)
    # print(iris_test5_results)
    iris_train6_class_type_dict = create_class_type_dict(iris_training6,iris_classes)
    iris_test6_results = calculate_feature_product(iris_testing6, iris_train6_likehood_table, iris_train6_class_type_dict)
    # print(iris_test6_results)
    iris_train7_class_type_dict = create_class_type_dict(iris_training7,iris_classes)
    iris_test7_results = calculate_feature_product(iris_testing7, iris_train7_likehood_table, iris_train7_class_type_dict)
    # print(iris_test7_results)
    iris_train8_class_type_dict = create_class_type_dict(iris_training8,iris_classes)
    iris_test8_results = calculate_feature_product(iris_testing8, iris_train8_likehood_table, iris_train8_class_type_dict)
    # print(iris_test8_results)
    iris_train9_class_type_dict = create_class_type_dict(iris_training9,iris_classes)
    iris_test9_results = calculate_feature_product(iris_testing9, iris_train9_likehood_table, iris_train9_class_type_dict)
    # print(iris_test9_results)
    iris_train10_class_type_dict = create_class_type_dict(iris_training10,iris_classes)
    iris_test10_results = calculate_feature_product(iris_testing10, iris_train10_likehood_table, iris_train10_class_type_dict)
    # print(iris_test10_results)

    
    soy_train1_class_type_dict = create_class_type_dict(soy_training1,soy_classes)
    soy_test1_results = calculate_feature_product(soy_testing1, soy_train1_likehood_table, soy_train1_class_type_dict)
    # print("--------------------------------")
    # print("RESULTS")
    # print(soy_test1_results)
    soy_train2_class_type_dict = create_class_type_dict(soy_training2,soy_classes)
    soy_test2_results = calculate_feature_product(soy_testing2, soy_train2_likehood_table, soy_train2_class_type_dict)
    soy_train3_class_type_dict = create_class_type_dict(soy_training3,soy_classes)
    soy_test3_results = calculate_feature_product(soy_testing3, soy_train3_likehood_table, soy_train3_class_type_dict)
    soy_train4_class_type_dict = create_class_type_dict(soy_training4,soy_classes)
    soy_test4_results = calculate_feature_product(soy_testing4, soy_train4_likehood_table, soy_train4_class_type_dict)
    soy_train5_class_type_dict = create_class_type_dict(soy_training5,soy_classes)
    soy_test5_results = calculate_feature_product(soy_testing5, soy_train5_likehood_table, soy_train5_class_type_dict)
    soy_train6_class_type_dict = create_class_type_dict(soy_training6,soy_classes)
    soy_test6_results = calculate_feature_product(soy_testing6, soy_train6_likehood_table, soy_train6_class_type_dict)
    soy_train7_class_type_dict = create_class_type_dict(soy_training7,soy_classes)
    soy_test7_results = calculate_feature_product(soy_testing7, soy_train7_likehood_table, soy_train7_class_type_dict)
    soy_train8_class_type_dict = create_class_type_dict(soy_training8,soy_classes)
    soy_test8_results = calculate_feature_product(soy_testing8, soy_train8_likehood_table, soy_train8_class_type_dict)
    soy_train9_class_type_dict = create_class_type_dict(soy_training9,soy_classes)
    soy_test9_results = calculate_feature_product(soy_testing9, soy_train9_likehood_table, soy_train9_class_type_dict)
    soy_train10_class_type_dict = create_class_type_dict(soy_training10,soy_classes)
    soy_test10_results = calculate_feature_product(soy_testing10, soy_train10_likehood_table, soy_train10_class_type_dict)


    # Perform classification on noise test sets
    cancer_noise_train1_class_type_dict = create_class_type_dict(cancer_noise_training1,cancer_classes)
    cancer_noise_test1_results = calculate_feature_product(cancer_noise_testing1, cancer_noise_train1_likehood_table, cancer_noise_train1_class_type_dict)
    # print("--------------------------------")
    # print("RESULTS")
    # print(cancer_noise_test1_results)
    cancer_noise_train2_class_type_dict = create_class_type_dict(cancer_noise_training2,cancer_classes)
    cancer_noise_test2_results = calculate_feature_product(cancer_noise_testing2, cancer_noise_train2_likehood_table, cancer_noise_train2_class_type_dict)
    cancer_noise_train3_class_type_dict = create_class_type_dict(cancer_noise_training3,cancer_classes)
    cancer_noise_test3_results = calculate_feature_product(cancer_noise_testing3, cancer_noise_train3_likehood_table, cancer_noise_train3_class_type_dict)
    cancer_noise_train4_class_type_dict = create_class_type_dict(cancer_noise_training4,cancer_classes)
    cancer_noise_test4_results = calculate_feature_product(cancer_noise_testing4, cancer_noise_train4_likehood_table, cancer_noise_train4_class_type_dict)
    cancer_noise_train5_class_type_dict = create_class_type_dict(cancer_noise_training5,cancer_classes)
    cancer_noise_test5_results = calculate_feature_product(cancer_noise_testing5, cancer_noise_train5_likehood_table, cancer_noise_train5_class_type_dict)
    cancer_noise_train6_class_type_dict = create_class_type_dict(cancer_noise_training6,cancer_classes)
    cancer_noise_test6_results = calculate_feature_product(cancer_noise_testing6, cancer_noise_train6_likehood_table, cancer_noise_train6_class_type_dict)
    cancer_noise_train7_class_type_dict = create_class_type_dict(cancer_noise_training7,cancer_classes)
    cancer_noise_test7_results = calculate_feature_product(cancer_noise_testing7, cancer_noise_train7_likehood_table, cancer_noise_train7_class_type_dict)
    cancer_noise_train8_class_type_dict = create_class_type_dict(cancer_noise_training8,cancer_classes)
    cancer_noise_test8_results = calculate_feature_product(cancer_noise_testing8, cancer_noise_train8_likehood_table, cancer_noise_train8_class_type_dict)
    cancer_noise_train9_class_type_dict = create_class_type_dict(cancer_noise_training9,cancer_classes)
    cancer_noise_test9_results = calculate_feature_product(cancer_noise_testing9, cancer_noise_train9_likehood_table, cancer_noise_train9_class_type_dict)
    cancer_noise_train10_class_type_dict = create_class_type_dict(cancer_noise_training10,cancer_classes)
    cancer_noise_test10_results = calculate_feature_product(cancer_noise_testing10, cancer_noise_train10_likehood_table, cancer_noise_train10_class_type_dict)

    glass_noise_train1_class_type_dict = create_class_type_dict(glass_noise_training1,glass_classes)
    glass_noise_test1_results = calculate_feature_product(glass_noise_testing1, glass_noise_train1_likehood_table, glass_noise_train1_class_type_dict)
    print("--------------------------------")
    print("TEST1")
    count = 0
    for row_idx, row in glass_noise_test1_results.iterrows():
        if int(row['class']) == int(row['Classifier']):
            count += 1
    print(count,"out of",glass_noise_test1_results.shape[0])
    glass_noise_train2_class_type_dict = create_class_type_dict(glass_noise_training2,glass_classes)
    glass_noise_test2_results = calculate_feature_product(glass_noise_testing2, glass_noise_train2_likehood_table, glass_noise_train2_class_type_dict)
    print("TEST 2")
    count = 0
    for row_idx, row in glass_noise_test2_results.iterrows():
        if int(row['class']) == int(row['Classifier']):
            count += 1
    print(count,"out of",glass_noise_test2_results.shape[0])
    glass_noise_train3_class_type_dict = create_class_type_dict(glass_noise_training3,glass_classes)
    glass_noise_test3_results = calculate_feature_product(glass_noise_testing3, glass_noise_train3_likehood_table, glass_noise_train3_class_type_dict)
    print("TEST3")
    count = 0
    for row_idx, row in glass_noise_test3_results.iterrows():
        if int(row['class']) == int(row['Classifier']):
            count += 1
    print(count,"out of",glass_noise_test3_results.shape[0])
    glass_noise_train4_class_type_dict = create_class_type_dict(glass_noise_training4,glass_classes)
    glass_noise_test4_results = calculate_feature_product(glass_noise_testing4, glass_noise_train4_likehood_table, glass_noise_train4_class_type_dict)
    print("TEST4")
    count = 0
    for row_idx, row in glass_noise_test4_results.iterrows():
        if int(row['class']) == int(row['Classifier']):
            count += 1
    print(count,"out of",glass_noise_test4_results.shape[0])
    glass_noise_train5_class_type_dict = create_class_type_dict(glass_noise_training5,glass_classes)
    glass_noise_test5_results = calculate_feature_product(glass_noise_testing5, glass_noise_train5_likehood_table, glass_noise_train5_class_type_dict)
    print("TEST5")
    count = 0
    for row_idx, row in glass_noise_test5_results.iterrows():
        if int(row['class']) == int(row['Classifier']):
            count += 1
    print(count,"out of",glass_noise_test5_results.shape[0])
    glass_noise_train6_class_type_dict = create_class_type_dict(glass_noise_training6,glass_classes)
    glass_noise_test6_results = calculate_feature_product(glass_noise_testing6, glass_noise_train6_likehood_table, glass_noise_train6_class_type_dict)
    print("TEST6")
    count = 0
    for row_idx, row in glass_noise_test6_results.iterrows():
        if int(row['class']) == int(row['Classifier']):
            count += 1
    print(count,"out of",glass_noise_test6_results.shape[0])
    glass_noise_train7_class_type_dict = create_class_type_dict(glass_noise_training7,glass_classes)
    glass_noise_test7_results = calculate_feature_product(glass_noise_testing7, glass_noise_train7_likehood_table, glass_noise_train7_class_type_dict)
    print("TEST7")
    count = 0
    for row_idx, row in glass_noise_test7_results.iterrows():
        if int(row['class']) == int(row['Classifier']):
            count += 1
    print(count,"out of",glass_noise_test7_results.shape[0])
    glass_noise_train8_class_type_dict = create_class_type_dict(glass_noise_training8,glass_classes)
    glass_noise_test8_results = calculate_feature_product(glass_noise_testing8, glass_noise_train8_likehood_table, glass_noise_train8_class_type_dict)
    print("TEST8")
    count = 0
    for row_idx, row in glass_noise_test8_results.iterrows():
        if int(row['class']) == int(row['Classifier']):
            count += 1
    print(count,"out of",glass_noise_test8_results.shape[0])
    glass_noise_train9_class_type_dict = create_class_type_dict(glass_noise_training9,glass_classes)
    glass_noise_test9_results = calculate_feature_product(glass_noise_testing9, glass_noise_train9_likehood_table, glass_noise_train9_class_type_dict)
    print("TEST9")
    count = 0
    for row_idx, row in glass_noise_test9_results.iterrows():
        if int(row['class']) == int(row['Classifier']):
            count += 1
    print(count,"out of",glass_noise_test9_results.shape[0])
    glass_noise_train10_class_type_dict = create_class_type_dict(glass_noise_training10,glass_classes)
    glass_noise_test10_results = calculate_feature_product(glass_noise_testing10, glass_noise_train10_likehood_table, glass_noise_train10_class_type_dict)
    print("TEST10")
    count = 0
    for row_idx, row in glass_noise_test10_results.iterrows():
        if int(row['class']) == int(row['Classifier']):
            count += 1
    print(count,"out of",glass_noise_test10_results.shape[0])

    votes_noise_train1_class_type_dict = create_class_type_dict(votes_noise_training1,votes_classes)
    votes_noise_test1_results = calculate_feature_product(votes_noise_testing1, votes_noise_train1_likehood_table, votes_noise_train1_class_type_dict)
    # print("--------------------------------")
    # print("RESULTS")
    # print(votes_noise_test1_results)
    votes_noise_train2_class_type_dict = create_class_type_dict(votes_noise_training2,votes_classes)
    votes_noise_test2_results = calculate_feature_product(votes_noise_testing2, votes_noise_train2_likehood_table, votes_noise_train2_class_type_dict)
    votes_noise_train3_class_type_dict = create_class_type_dict(votes_noise_training3,votes_classes)
    votes_noise_test3_results = calculate_feature_product(votes_noise_testing3, votes_noise_train3_likehood_table, votes_noise_train3_class_type_dict)
    votes_noise_train4_class_type_dict = create_class_type_dict(votes_noise_training4,votes_classes)
    votes_noise_test4_results = calculate_feature_product(votes_noise_testing4, votes_noise_train4_likehood_table, votes_noise_train4_class_type_dict)
    votes_noise_train5_class_type_dict = create_class_type_dict(votes_noise_training5,votes_classes)
    votes_noise_test5_results = calculate_feature_product(votes_noise_testing5, votes_noise_train5_likehood_table, votes_noise_train5_class_type_dict)
    votes_noise_train6_class_type_dict = create_class_type_dict(votes_noise_training6,votes_classes)
    votes_noise_test6_results = calculate_feature_product(votes_noise_testing6, votes_noise_train6_likehood_table, votes_noise_train6_class_type_dict)
    votes_noise_train7_class_type_dict = create_class_type_dict(votes_noise_training7,votes_classes)
    votes_noise_test7_results = calculate_feature_product(votes_noise_testing7, votes_noise_train7_likehood_table, votes_noise_train7_class_type_dict)
    votes_noise_train8_class_type_dict = create_class_type_dict(votes_noise_training8,votes_classes)
    votes_noise_test8_results = calculate_feature_product(votes_noise_testing8, votes_noise_train8_likehood_table, votes_noise_train8_class_type_dict)
    votes_noise_train9_class_type_dict = create_class_type_dict(votes_noise_training9,votes_classes)
    votes_noise_test9_results = calculate_feature_product(votes_noise_testing9, votes_noise_train9_likehood_table, votes_noise_train9_class_type_dict)
    votes_noise_train10_class_type_dict = create_class_type_dict(votes_noise_training10,votes_classes)
    votes_noise_test10_results = calculate_feature_product(votes_noise_testing10, votes_noise_train10_likehood_table, votes_noise_train10_class_type_dict)

    iris_noise_train1_class_type_dict = create_class_type_dict(iris_noise_training1,iris_classes)
    iris_noise_test1_results = calculate_feature_product(iris_noise_testing1, iris_noise_train1_likehood_table, iris_noise_train1_class_type_dict)
    # print("--------------------------------")
    # print("RESULTS")
    # print(iris_noise_test1_results)
    iris_noise_train2_class_type_dict = create_class_type_dict(iris_noise_training2,iris_classes)
    iris_noise_test2_results = calculate_feature_product(iris_noise_testing2, iris_noise_train2_likehood_table, iris_noise_train2_class_type_dict)
    # print(iris_noise_test2_results)
    iris_noise_train3_class_type_dict = create_class_type_dict(iris_noise_training3,iris_classes)
    iris_noise_test3_results = calculate_feature_product(iris_noise_testing3, iris_noise_train3_likehood_table, iris_noise_train3_class_type_dict)
    # print(iris_noise_test3_results)
    iris_noise_train4_class_type_dict = create_class_type_dict(iris_noise_training4,iris_classes)
    iris_noise_test4_results = calculate_feature_product(iris_noise_testing4, iris_noise_train4_likehood_table, iris_noise_train4_class_type_dict)
    # print(iris_noise_test4_results)
    iris_noise_train5_class_type_dict = create_class_type_dict(iris_noise_training5,iris_classes)
    iris_noise_test5_results = calculate_feature_product(iris_noise_testing5, iris_noise_train5_likehood_table, iris_noise_train5_class_type_dict)
    # print(iris_noise_test5_results)
    iris_noise_train6_class_type_dict = create_class_type_dict(iris_noise_training6,iris_classes)
    iris_noise_test6_results = calculate_feature_product(iris_noise_testing6, iris_noise_train6_likehood_table, iris_noise_train6_class_type_dict)
    # print(iris_noise_test6_results)
    iris_noise_train7_class_type_dict = create_class_type_dict(iris_noise_training7,iris_classes)
    iris_noise_test7_results = calculate_feature_product(iris_noise_testing7, iris_noise_train7_likehood_table, iris_noise_train7_class_type_dict)
    # print(iris_noise_test7_results)
    iris_noise_train8_class_type_dict = create_class_type_dict(iris_noise_training8,iris_classes)
    iris_noise_test8_results = calculate_feature_product(iris_noise_testing8, iris_noise_train8_likehood_table, iris_noise_train8_class_type_dict)
    # print(iris_noise_test8_results)
    iris_noise_train9_class_type_dict = create_class_type_dict(iris_noise_training9,iris_classes)
    iris_noise_test9_results = calculate_feature_product(iris_noise_testing9, iris_noise_train9_likehood_table, iris_noise_train9_class_type_dict)
    # print(iris_noise_test9_results)
    iris_noise_train10_class_type_dict = create_class_type_dict(iris_noise_training10,iris_classes)
    iris_noise_test10_results = calculate_feature_product(iris_noise_testing10, iris_noise_train10_likehood_table, iris_noise_train10_class_type_dict)
    # print(iris_noise_test10_results)

    
    soy_noise_train1_class_type_dict = create_class_type_dict(soy_noise_training1,soy_classes)
    soy_noise_test1_results = calculate_feature_product(soy_noise_testing1, soy_noise_train1_likehood_table, soy_noise_train1_class_type_dict)
    # print("--------------------------------")
    # print("RESULTS")
    # print(soy_noise_test1_results)
    soy_noise_train2_class_type_dict = create_class_type_dict(soy_noise_training2,soy_classes)
    soy_noise_test2_results = calculate_feature_product(soy_noise_testing2, soy_noise_train2_likehood_table, soy_noise_train2_class_type_dict)
    soy_noise_train3_class_type_dict = create_class_type_dict(soy_noise_training3,soy_classes)
    soy_noise_test3_results = calculate_feature_product(soy_noise_testing3, soy_noise_train3_likehood_table, soy_noise_train3_class_type_dict)
    soy_noise_train4_class_type_dict = create_class_type_dict(soy_noise_training4,soy_classes)
    soy_noise_test4_results = calculate_feature_product(soy_noise_testing4, soy_noise_train4_likehood_table, soy_noise_train4_class_type_dict)
    soy_noise_train5_class_type_dict = create_class_type_dict(soy_noise_training5,soy_classes)
    soy_noise_test5_results = calculate_feature_product(soy_noise_testing5, soy_noise_train5_likehood_table, soy_noise_train5_class_type_dict)
    soy_noise_train6_class_type_dict = create_class_type_dict(soy_noise_training6,soy_classes)
    soy_noise_test6_results = calculate_feature_product(soy_noise_testing6, soy_noise_train6_likehood_table, soy_noise_train6_class_type_dict)
    soy_noise_train7_class_type_dict = create_class_type_dict(soy_noise_training7,soy_classes)
    soy_noise_test7_results = calculate_feature_product(soy_noise_testing7, soy_noise_train7_likehood_table, soy_noise_train7_class_type_dict)
    soy_noise_train8_class_type_dict = create_class_type_dict(soy_noise_training8,soy_classes)
    soy_noise_test8_results = calculate_feature_product(soy_noise_testing8, soy_noise_train8_likehood_table, soy_noise_train8_class_type_dict)
    soy_noise_train9_class_type_dict = create_class_type_dict(soy_noise_training9,soy_classes)
    soy_noise_test9_results = calculate_feature_product(soy_noise_testing9, soy_noise_train9_likehood_table, soy_noise_train9_class_type_dict)
    soy_noise_train10_class_type_dict = create_class_type_dict(soy_noise_training10,soy_classes)
    soy_noise_test10_results = calculate_feature_product(soy_noise_testing10, soy_noise_train10_likehood_table, soy_noise_train10_class_type_dict)

    print("CALCULATING LOSS FUNCTIONS")
    # Loss functions on original sets
    cancer_loss1 = calculate_loss_function(cancer_test1_results,cancer_classes)
    cancer_loss2 = calculate_loss_function(cancer_test2_results,cancer_classes)
    cancer_loss3 = calculate_loss_function(cancer_test3_results,cancer_classes)
    cancer_loss4 = calculate_loss_function(cancer_test4_results,cancer_classes)
    cancer_loss5 = calculate_loss_function(cancer_test5_results,cancer_classes)
    cancer_loss6 = calculate_loss_function(cancer_test6_results,cancer_classes)
    cancer_loss7 = calculate_loss_function(cancer_test7_results,cancer_classes)
    cancer_loss8 = calculate_loss_function(cancer_test8_results,cancer_classes)
    cancer_loss9 = calculate_loss_function(cancer_test9_results,cancer_classes)
    cancer_loss10 = calculate_loss_function(cancer_test10_results,cancer_classes)

    glass_loss1 = calculate_loss_function(glass_test1_results,glass_classes)
    glass_loss2 = calculate_loss_function(glass_test2_results,glass_classes)
    glass_loss3 = calculate_loss_function(glass_test3_results,glass_classes)
    glass_loss4 = calculate_loss_function(glass_test4_results,glass_classes)
    glass_loss5 = calculate_loss_function(glass_test5_results,glass_classes)
    glass_loss6 = calculate_loss_function(glass_test6_results,glass_classes)
    glass_loss7 = calculate_loss_function(glass_test7_results,glass_classes)
    glass_loss8 = calculate_loss_function(glass_test8_results,glass_classes)
    glass_loss9 = calculate_loss_function(glass_test9_results,glass_classes)
    glass_loss10 = calculate_loss_function(glass_test10_results,glass_classes)

    votes_loss1 = calculate_loss_function(votes_test1_results,votes_classes)
    votes_loss2 = calculate_loss_function(votes_test2_results,votes_classes)
    votes_loss3 = calculate_loss_function(votes_test3_results,votes_classes)
    votes_loss4 = calculate_loss_function(votes_test4_results,votes_classes)
    votes_loss5 = calculate_loss_function(votes_test5_results,votes_classes)
    votes_loss6 = calculate_loss_function(votes_test6_results,votes_classes)
    votes_loss7 = calculate_loss_function(votes_test7_results,votes_classes)
    votes_loss8 = calculate_loss_function(votes_test8_results,votes_classes)
    votes_loss9 = calculate_loss_function(votes_test9_results,votes_classes)
    votes_loss10 = calculate_loss_function(votes_test10_results,votes_classes)

    iris_loss1 = calculate_loss_function(iris_test1_results,iris_classes)
    iris_loss2 = calculate_loss_function(iris_test2_results,iris_classes)
    iris_loss3 = calculate_loss_function(iris_test3_results,iris_classes)
    iris_loss4 = calculate_loss_function(iris_test4_results,iris_classes)
    iris_loss5 = calculate_loss_function(iris_test5_results,iris_classes)
    iris_loss6 = calculate_loss_function(iris_test6_results,iris_classes)
    iris_loss7 = calculate_loss_function(iris_test7_results,iris_classes)
    iris_loss8 = calculate_loss_function(iris_test8_results,iris_classes)
    iris_loss9 = calculate_loss_function(iris_test9_results,iris_classes)
    iris_loss10 = calculate_loss_function(iris_test10_results,iris_classes)

    soy_loss1 = calculate_loss_function(soy_test1_results,soy_classes)
    soy_loss2 = calculate_loss_function(soy_test2_results,soy_classes)
    soy_loss3 = calculate_loss_function(soy_test3_results,soy_classes)
    soy_loss4 = calculate_loss_function(soy_test4_results,soy_classes)
    soy_loss5 = calculate_loss_function(soy_test5_results,soy_classes)
    soy_loss6 = calculate_loss_function(soy_test6_results,soy_classes)
    soy_loss7 = calculate_loss_function(soy_test7_results,soy_classes)
    soy_loss8 = calculate_loss_function(soy_test8_results,soy_classes)
    soy_loss9 = calculate_loss_function(soy_test9_results,soy_classes)
    soy_loss10 = calculate_loss_function(soy_test10_results,soy_classes)

    # Loss functions on noise sets
    cancer_noise_loss1 = calculate_loss_function(cancer_noise_test1_results,cancer_classes)
    cancer_noise_loss2 = calculate_loss_function(cancer_noise_test2_results,cancer_classes)
    cancer_noise_loss3 = calculate_loss_function(cancer_noise_test3_results,cancer_classes)
    cancer_noise_loss4 = calculate_loss_function(cancer_noise_test4_results,cancer_classes)
    cancer_noise_loss5 = calculate_loss_function(cancer_noise_test5_results,cancer_classes)
    cancer_noise_loss6 = calculate_loss_function(cancer_noise_test6_results,cancer_classes)
    cancer_noise_loss7 = calculate_loss_function(cancer_noise_test7_results,cancer_classes)
    cancer_noise_loss8 = calculate_loss_function(cancer_noise_test8_results,cancer_classes)
    cancer_noise_loss9 = calculate_loss_function(cancer_noise_test9_results,cancer_classes)
    cancer_noise_loss10 = calculate_loss_function(cancer_noise_test10_results,cancer_classes)

    glass_noise_loss1 = calculate_loss_function(glass_noise_test1_results,glass_classes)
    glass_noise_loss2 = calculate_loss_function(glass_noise_test2_results,glass_classes)
    glass_noise_loss3 = calculate_loss_function(glass_noise_test3_results,glass_classes)
    glass_noise_loss4 = calculate_loss_function(glass_noise_test4_results,glass_classes)
    glass_noise_loss5 = calculate_loss_function(glass_noise_test5_results,glass_classes)
    glass_noise_loss6 = calculate_loss_function(glass_noise_test6_results,glass_classes)
    glass_noise_loss7 = calculate_loss_function(glass_noise_test7_results,glass_classes)
    glass_noise_loss8 = calculate_loss_function(glass_noise_test8_results,glass_classes)
    glass_noise_loss9 = calculate_loss_function(glass_noise_test9_results,glass_classes)
    glass_noise_loss10 = calculate_loss_function(glass_noise_test10_results,glass_classes)

    votes_noise_loss1 = calculate_loss_function(votes_noise_test1_results,votes_classes)
    votes_noise_loss2 = calculate_loss_function(votes_noise_test2_results,votes_classes)
    votes_noise_loss3 = calculate_loss_function(votes_noise_test3_results,votes_classes)
    votes_noise_loss4 = calculate_loss_function(votes_noise_test4_results,votes_classes)
    votes_noise_loss5 = calculate_loss_function(votes_noise_test5_results,votes_classes)
    votes_noise_loss6 = calculate_loss_function(votes_noise_test6_results,votes_classes)
    votes_noise_loss7 = calculate_loss_function(votes_noise_test7_results,votes_classes)
    votes_noise_loss8 = calculate_loss_function(votes_noise_test8_results,votes_classes)
    votes_noise_loss9 = calculate_loss_function(votes_noise_test9_results,votes_classes)
    votes_noise_loss10 = calculate_loss_function(votes_noise_test10_results,votes_classes)

    iris_noise_loss1 = calculate_loss_function(iris_noise_test1_results,iris_classes)
    iris_noise_loss2 = calculate_loss_function(iris_noise_test2_results,iris_classes)
    iris_noise_loss3 = calculate_loss_function(iris_noise_test3_results,iris_classes)
    iris_noise_loss4 = calculate_loss_function(iris_noise_test4_results,iris_classes)
    iris_noise_loss5 = calculate_loss_function(iris_noise_test5_results,iris_classes)
    iris_noise_loss6 = calculate_loss_function(iris_noise_test6_results,iris_classes)
    iris_noise_loss7 = calculate_loss_function(iris_noise_test7_results,iris_classes)
    iris_noise_loss8 = calculate_loss_function(iris_noise_test8_results,iris_classes)
    iris_noise_loss9 = calculate_loss_function(iris_noise_test9_results,iris_classes)
    iris_noise_loss10 = calculate_loss_function(iris_noise_test10_results,iris_classes)

    soy_noise_loss1 = calculate_loss_function(soy_noise_test1_results,soy_classes)
    soy_noise_loss2 = calculate_loss_function(soy_noise_test2_results,soy_classes)
    soy_noise_loss3 = calculate_loss_function(soy_noise_test3_results,soy_classes)
    soy_noise_loss4 = calculate_loss_function(soy_noise_test4_results,soy_classes)
    soy_noise_loss5 = calculate_loss_function(soy_noise_test5_results,soy_classes)
    soy_noise_loss6 = calculate_loss_function(soy_noise_test6_results,soy_classes)
    soy_noise_loss7 = calculate_loss_function(soy_noise_test7_results,soy_classes)
    soy_noise_loss8 = calculate_loss_function(soy_noise_test8_results,soy_classes)
    soy_noise_loss9 = calculate_loss_function(soy_noise_test9_results,soy_classes)
    soy_noise_loss10 = calculate_loss_function(soy_noise_test10_results,soy_classes)
