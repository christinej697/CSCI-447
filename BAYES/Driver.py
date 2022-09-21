from pickle import NONE
import pandas as pd
import numpy as np
# this function will stratify the dataset into 10 even size datasets that are proportional to the origional data set's classes
#split into 10 even sections, counts of each class, frequency of attribute
#returns 10 lists of even lenght of records that are each are proportional to the origional data set's classes
# def stratify(data,class_col): # file data, # of class column in data
#     ten_strata = [[],[],[],[],[],[],[],[],[],[]]

#     class_data = {} #dictionary of class_name:data of all records for that class
#     class_names = data["Class"].unique()
#     for i in data:
#         columns = i.replace('\n','').split(",")
#         class_name = columns[class_col]
#         try:
#             data = class_data[class_name]
#         except:
#             data = ""
#         class_data.update([(class_name,data+i)]) #add the data in i to our class (class is represented by the key)

#     class_frequency = {}  # dictionary of class_name: number of times that class occurs in data
#     for key in class_data: #find frequency of each class
#         counts = class_data[key].count('\n')
#         class_frequency.update([(key,counts)])

#     for key in class_data: #assign records to each strata
#         #fix rounding
#         strata_subsize = class_frequency[key]/10
#         for i in range(10):
#             start_range = int(i * strata_subsize)
#             end_range = int(i * strata_subsize + strata_subsize)
#             records = class_data[key].split('\n')
#             ten_strata[i]+=(records[start_range:end_range])


def stratify(dataset):
    for rowIndex, row in dataset.iterrows(): #iterate over rows
        for columnIndex, value in row.items():
            print(value, end="\t")
            print()

   
            
            
        
    

        
    
        
    

def main():
    ### read in the file
    file_name = "breast-cancer-wisconsin.data"
    dataset = pd.read_csv(file_name, sep=",")

    # drop all the ? cells
    dataset = dataset.replace("?", np.NaN)
    dataset = dataset.dropna()

    # sorted data by class
    dataset.columns = ["Sample code number", "Uniformity of Clump Thickness", "Uniformity of Cell Size", "Cell Shape", "Marginal Adhesion", "Single Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses", "Class"]
    dataset = dataset.sort_values(by=['Class'])
    stratify_and_fold(dataset)

main()