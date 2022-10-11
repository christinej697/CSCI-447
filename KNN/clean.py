from hashlib import new
import math
from timeit import repeat
import pandas as pd
import numpy as np
import random as rd
# import matplotlib.pyplot as plt

# step 2 generate the centroids randomly
def generate_centroids(data, k):
    centroids = data.iloc[np.random.choice(np.arange(len(data)), k, False)]
    return centroids

 # function to calculate the euclidean distance between a training instance and a test instance
def euclidean_distance(train_row: pd.Series, test_row: pd.Series) -> int:
    euclidean_distance = 0
        # Get sum of attribute distances
        # print(train_row)
        # print()
        # print(test_row)
        # print()
    for i in range(0,train_row.shape[0]-1):
            # print(i)
            # print(train_row[i])
            # [print(test_row[i])]
        euclidean_distance += pow((train_row[i] - test_row[i]),2)
            # sys.exit(0)
        # Return the square root of the sum
    return math.sqrt(euclidean_distance)

# step assign all the points to the closet cluster centroid
def generate_clusters(data, k, centroids):
    # to store teh distance 
    temp_data = data.copy()
    index = 1
    for c_row_idx, c_row in centroids.iterrows():
        eculidean_dists = []
        for d_row_idx, d_row in data.iterrows():
            ed = euclidean_distance(c_row, d_row)
            eculidean_dists.append(ed)
        print(len(eculidean_dists))
        temp_data[index] = eculidean_dists
        index = index + 1
    # create a list to store clusters 
    cluster = []
    for ed_index, ed_row in data.iterrows():
        # set the first position of the row to be the min
        min_distance = ed_row[1]
        cluster_num = 1
        for i in range(k):
            if ed_row[i+1] < min_distance:
                cluster_num = cluster_num + 1
        cluster.append(cluster_num)
    temp_data["Cluster"] = cluster
    ##print(temp_data)
    return temp_data

def k_mean_clustering(sub_data, k, centeroids, labels):
    print("INTO CLUSTERING")
    stop = 1
    print(centroids)
    old_centeroids = centeroids.reset_index(drop=True)
    old_data = sub_data.copy()
    while (stop != 0):
         # to store the distance 
        temp_data = sub_data.copy()
        index = 1
        for c_row_idx, c_row in old_centeroids.iterrows():
            eculidean_dists = []
            for d_row_idx, d_row in sub_data.iterrows():
                ed = euclidean_distance(c_row, d_row)
                eculidean_dists.append(ed)
            temp_data[index] = eculidean_dists
            index = index + 1
        # create a list to store clusters 
        cluster = []
        for ed_index, ed_row in temp_data.iterrows():
            # set the first position of the row to be the min
            min_distance = ed_row[1]
            cluster_num = 1
            for i in range(0,k):
                if ed_row[i+1] < min_distance:
                    cluster_num = i + 1
            cluster.append(cluster_num)
        temp_data["Cluster"] = cluster
        # raise SystemExit(0)
        print("+++++++++++++old centriods++++++++++++++++\n")
        print(old_centeroids)
        # old_ = generate_clusters(sub_data, k, centroids)
        print("--------------finding new centriods-----------\n")
        new_centeroids = temp_data.groupby(['Cluster']).mean()[labels].reset_index(drop=True)
        print(new_centeroids)
       
        eds = []
        for row_idx, c_row in new_centeroids.iterrows():
            ed = euclidean_distance(new_centeroids.iloc[row_idx], old_centeroids.iloc[row_idx])
            eds.append(ed)
        stop = sum(eds)
        print(stop)
        print("*************************************************")
        old_centeroids = new_centeroids.copy()
    
    
       

    #new_centroids = []
    # for i in range(1, k+1):
    #     cluster_dict[i] = data[data['Cluster']==i].reset_index(drop=True)
    #     index = rd.randint(0, len(cluster_dict[i]))
    #     row = cluster_dict[i].iloc[[index]]
    #     new_centroids.append(row)
    # new_centroids = pd.concat(new_centroids)
    # print(new_centroids)
    
    # for j in range(len(new_centroids)):
    #     euclidean_distance(new_centroids[j], centeroids[j])

    # print(cluster_dict)
   

    # print(new_centroids)
    # centroids_stop = 1
    # for row_index, c_row in new_centroids.iterrows():
    #     centroids_stop  = euclidean_distance(new_centroids[row_index], centeroids[row_index])
    #     print(centeroids)
        



    
data = pd.read_csv("breast-cancer-wisconsin-cleaned.txt", sep=",", header=None)
print(data.head)
cancer_labels = ["Sample code number", "Uniformity of Clump Thickness", "Uniformity of Cell Size", "Cell Shape", "Marginal Adhesion", "Single Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses", "class"]
data.columns = cancer_labels
clean_labels= ["Uniformity of Clump Thickness", "Uniformity of Cell Size", "Cell Shape", "Marginal Adhesion", "Single Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses"]
df = pd.DataFrame(data)
sub_data = df[["Uniformity of Clump Thickness", "Uniformity of Cell Size", "Cell Shape", "Marginal Adhesion", "Single Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses"]]
k = 3
centroids = generate_centroids(sub_data, 3)
print(centroids)
#print(centroids.iloc[1])
k_mean_clustering(sub_data, k, centroids, clean_labels)