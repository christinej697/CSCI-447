# Class to implement to KNN
import math
from turtle import pos
import pandas as pd
import numpy as np
from typing import Tuple
from statistics import mode
import sys

class KNN:
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

        forestfires_df = pd.read_csv("forestfires.csv", sep=",")

        
        print("STRATIFYING DATA AND CREATING TUNING & FOLDS...")
        # Create training and testing dataframes for classification data, as well as the tuning dataframe
        cancer_training1,cancer_testing1,cancer_training2,cancer_testing2,cancer_training3,cancer_testing3,cancer_training4,cancer_testing4,cancer_training5,cancer_testing5,cancer_training6,cancer_testing6,cancer_training7,cancer_testing7,cancer_training8,cancer_testing8,cancer_training9,cancer_testing9,cancer_training10,cancer_testing10,cancer_tuning = self.stratify_and_fold_classification(cancer_df)
        # glass_training1,glass_testing1,glass_training2,glass_testing2,glass_training3,glass_testing3,glass_training4,glass_testing4,glass_training5,glass_testing5,glass_training6,glass_testing6,glass_training7,glass_testing7,glass_training8,glass_testing8,glass_training9,glass_testing9,glass_training10,glass_testing10,glass_tuning = self.stratify_and_fold_classification(glass_df)
        # soy_training1,soy_testing1,soy_training2,soy_testing2,soy_training3,soy_testing3,soy_training4,soy_testing4,soy_training5,soy_testing5,soy_training6,soy_testing6,soy_training7,soy_testing7,soy_training8,soy_testing8,soy_training9,soy_testing9,soy_training10,soy_testing10,soy_tuning = self.stratify_and_fold_classification(soy_df)

        # Create training and testing dataframes for regression data
        abalone_training1,abalone_testing1,abalone_training2,abalone_testing2,abalone_training3,abalone_testing3,abalone_training4,abalone_testing4,abalone_training5,abalone_testing5,abalone_training6,abalone_testing6,abalone_training7,abalone_testing7,abalone_training8,abalone_testing8,abalone_training9,abalone_testing9,abalone_training10,abalone_testing10,abalone_tuning = self.stratify_and_fold_regression(abalone_df)
        machine_training1,machine_testing1,machine_training2,machine_testing2,machine_training3,machine_testing3,machine_training4,machine_testing4,machine_training5,machine_testing5,machine_training6,machine_testing6,machine_training7,machine_testing7,machine_training8,machine_testing8,machine_training9,machine_testing9,machine_training10,machine_testing10,machine_tuning = self.stratify_and_fold_regression(machine_df)
        # forestfires_training1,forestfires_testing1,forestfires_training2,forestfires_testing2,forestfires_training3,forestfires_testing3,forestfires_training4,forestfires_testing4,forestfires_training5,forestfires_testing5,forestfires_training6,forestfires_testing6,forestfires_training7,forestfires_testing7,forestfires_training8,forestfires_testing8,forestfires_training9,forestfires_testing9,forestfires_training10,forestfires_testing10,forestfires_tuning = self.stratify_and_fold_regression(forestfires_df)

        # apply KNN
        # abalone_knn1 = self.knn(abalone_training1, abalone_testing1, 1, "regression")
        # print(abalone_knn1)

        # abalone_knn1 = self.knn(abalone_training1, abalone_testing1, 2, "regression")
        # print(abalone_knn1)

        # abalone_knn1 = self.knn(abalone_training1, abalone_testing1, 3, "regression")
        # print(abalone_knn1)
        # abalone_knn1 = self.knn(abalone_training1, abalone_testing1, 4, "regression")
        # print(abalone_knn1)
        # abalone_knn1 = self.knn(abalone_training1, abalone_testing1, 5, "regression")
        # print(abalone_knn1)
        # machine_knn1 = self.knn(machine_training1, machine_testing1, 2, "regression")
        # print(machine_knn1)

        # apply edited KNN
        # abalone_eknn1 = self.knn(self.eknn(abalone_training1, 2, "regression"), abalone_testing1, 2, "regression")
        # print(abalone_eknn1)
        # abalone_knn1 = self.knn(abalone_training1, abalone_testing1, 2, "regression")
        # print(abalone_knn1)

        # test out classification value difference w/ KNN
        print(cancer_testing1)
        iris_knn = self.knn(cancer_training1, cancer_testing1, 3, "classification")
        print(iris_knn)

        
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
        print("Entering VDM...")
        # empty dictionary to hold all feature difference matrices
        diff_matrix_dict = {}
        classes = train_df.iloc[:,-1].unique()
        # for each feature in x
        for col_name, col_value in train_df.iteritems():
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
                        ci = train_df[train_df['col_name'] == vi].shape[0]
                        cia = train_df[(train_df['col_name'] == vi) & (train_df.iloc[:,-1] == c)].shape[0]
                        cj = train_df[train_df['col_name'] == vj].shape[0]
                        cja = train_df[(train_df['col_name'] == vj) & (train_df.iloc[:,-1] == c)].shape[0]
                        d_vi_vj += pow(abs((cia/ci) - (cja/cj)),2)
                    # put d(vi,vj) in the feature differences matrix
                    diff_matrix[idx_i,idx_j] = d_vi_vj
            # put the feature valu matrix into the dictionary
            diff_matrix_dict[col_name] = diff_matrix
        # return the dictionary of all feature difference matrices
        return diff_matrix_dict

    # function to handle categorical values

    # employ a plurality vote to determine the class
    def plurality_vote(self) -> None:
        pass
    
    def guassian_kernel(self) -> None:
        pass
    
    # function to perform knn on given training and test datasets
    def knn(self, train_df: pd.DataFrame, test_df:pd.DataFrame, k: int, version: str) -> pd.DataFrame:
        print("Entering KNN...")
        # print("TESTING")
        # print(test_df)
        # print(type(test_df))
        predictions = []
        if version == "classification":
            # get feature difference matrices
            diff_matrix_dict = self.value_difference_metric()
        # Loop through each instance in the testing dataset
        for test_row_index, test_row in test_df.iterrows():
            # Loop through each instance in the training set
            distances = []
            # print()
            # print("ROW")
            # print(test_row)
            for train_row_index, train_row in train_df.iterrows():
                # apply Euclidean distance funciton if regression
                if version == "regression":
                    # Get euclidean distance between current test instance and a given instance in test set
                    distances.append(self.euclidean_distance(train_row, test_row))
                # apply distance based on value difference metric if classification
                elif version == "classification":
                    d_x_y = 0
                    # iterate through all features in x
                    for col_name, col_value in train_row:
                        d_x_y += diff_matrix_dict[col_name][col_value][test_row[col_name]]
                    # Add distance between current test instance and a given instance in test set to distances array
                    distances.append(math.sqrt(d_x_y))
            # Add the returned distances onto the end of the training set
            train_df["Distance"] = distances
            # Find the min k distances in the training set
            sorted_df = train_df.sort_values("Distance")
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
                    k_classes = sorted_df.loc[sorted_df.index[i], "class"]
                predictions.append(mode(k_classes))
            # Move to the next test instance
        # Set the predictions to a column on the test data set
        test_df['KNN_Prediction'] = predictions
        # Return the test set with KNN Predictions appended
        return test_df

    # function to calculate the euclidean distance between a training instance and a test instance
    def euclidean_distance(self, train_row: pd.Series, test_row: pd.Series) -> int:
        euclidean_distance = 0
        # Get sum of attribute distances
        # print(train_row)
        # print()
        # print(test_row)
        # print()
        for i in range(1,train_row.shape[0]-1):
            # print(i)
            # print(train_row[i])
            # [print(test_row[i])]
            euclidean_distance += pow((train_row[i] - test_row[i]),2)
            # sys.exit(0)
        # Return the square root of the sum
        return math.sqrt(euclidean_distance)

    # function to perform edited k nearest neighbor on a given training dataset, using error removal method
    # returns reduced dataset
    def eknn(self, dataset: pd.DataFrame, k: int, version: str) -> pd.DataFrame:
        # run until points are not edited\
        performance_flag = True
        dataset = dataset.head(1500)
        while performance_flag:
            remove_points = []
            cnt = 0
            original_size = dataset.shape[0]
            indx = 1
            # loop through all points in E
            print("Length",original_size)
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
                classified_row = self.knn(df_no_xi, row_df, k, version)
                # Compare the classification returned with the stored class label
                if classified_row.iat[0,-1] != row[-1]:
                    remove_points.append(row_index)
                else:
                    cnt += 1
                # print("Row",indx,"is done")
                indx += 1
            # remove all points from E where the classifaction was incorrect
            print(remove_points)
            # print(dataset.loc[])
            # for i in range(len(remove_points)):
            dataset.drop(remove_points, inplace=True)
            # if no points removed from E this round, end loop and return
            if cnt == original_size:
                performance_flag = False
                print("FINITO")
                edited_dataset = dataset.copy()
            print("Original length:",original_size,"--------- CNT:",cnt)
        print(edited_dataset)
        print(edited_dataset.shape)
        return edited_dataset

    # function to perform k means clustering to use for knn
    def km_cluster(self) -> None:
        pass


knn = KNN()
knn.main()