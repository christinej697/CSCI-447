import pandas as pd
import numpy as np
import random

# create frequency tables for a given cancer data segment
def cancer_frequency(train_set):
    data_struct = [
        [train_set[(train_set['class'] == 2) & (train_set['Uniformity of Clump Thickness'] == 1)].shape[0],train_set[(train_set['class'] == 2) & (train_set['Uniformity of Cell Size'] == 1)].shape[0],train_set[(train_set['class'] == 2) & (train_set['Cell Shape'] == 1)].shape[0],train_set[(train_set['class'] == 2) & (train_set['Marginal Adhesion'] == 1)].shape[0],train_set[(train_set['class'] == 2) & (train_set['Single Epithelial Cell Size'] == 1)].shape[0],train_set[(train_set['class'] == 2) & (train_set['Bare Nuclei'] == '1')].shape[0],train_set[(train_set['class'] == 2) & (train_set['Bland Chromatin'] == 1)].shape[0],train_set[(train_set['class'] == 2) & (train_set['Normal Nucleoli'] == 1)].shape[0],train_set[(train_set['class'] == 2) & (train_set['Mitoses'] == 1)].shape[0]],
        [train_set[(train_set['class'] == 2) & (train_set['Uniformity of Clump Thickness'] == 2)].shape[0],train_set[(train_set['class'] == 2) & (train_set['Uniformity of Cell Size'] == 2)].shape[0],train_set[(train_set['class'] == 2) & (train_set['Cell Shape'] == 2)].shape[0],train_set[(train_set['class'] == 2) & (train_set['Marginal Adhesion'] == 2)].shape[0],train_set[(train_set['class'] == 2) & (train_set['Single Epithelial Cell Size'] == 2)].shape[0],train_set[(train_set['class'] == 2) & (train_set['Bare Nuclei'] == '2')].shape[0],train_set[(train_set['class'] == 2) & (train_set['Bland Chromatin'] == 2)].shape[0],train_set[(train_set['class'] == 2) & (train_set['Normal Nucleoli'] == 2)].shape[0],train_set[(train_set['class'] == 2) & (train_set['Mitoses'] == 2)].shape[0]],
        [train_set[(train_set['class'] == 2) & (train_set['Uniformity of Clump Thickness'] == 3)].shape[0],train_set[(train_set['class'] == 2) & (train_set['Uniformity of Cell Size'] == 3)].shape[0],train_set[(train_set['class'] == 2) & (train_set['Cell Shape'] == 3)].shape[0],train_set[(train_set['class'] == 2) & (train_set['Marginal Adhesion'] == 3)].shape[0],train_set[(train_set['class'] == 2) & (train_set['Single Epithelial Cell Size'] == 3)].shape[0],train_set[(train_set['class'] == 2) & (train_set['Bare Nuclei'] == '3')].shape[0],train_set[(train_set['class'] == 2) & (train_set['Bland Chromatin'] == 3)].shape[0],train_set[(train_set['class'] == 2) & (train_set['Normal Nucleoli'] == 3)].shape[0],train_set[(train_set['class'] == 2) & (train_set['Mitoses'] == 3)].shape[0]],
        [train_set[(train_set['class'] == 2) & (train_set['Uniformity of Clump Thickness'] == 4)].shape[0],train_set[(train_set['class'] == 2) & (train_set['Uniformity of Cell Size'] == 4)].shape[0],train_set[(train_set['class'] == 2) & (train_set['Cell Shape'] == 4)].shape[0],train_set[(train_set['class'] == 2) & (train_set['Marginal Adhesion'] == 4)].shape[0],train_set[(train_set['class'] == 2) & (train_set['Single Epithelial Cell Size'] == 4)].shape[0],train_set[(train_set['class'] == 2) & (train_set['Bare Nuclei'] == '4')].shape[0],train_set[(train_set['class'] == 2) & (train_set['Bland Chromatin'] == 4)].shape[0],train_set[(train_set['class'] == 2) & (train_set['Normal Nucleoli'] == 4)].shape[0],train_set[(train_set['class'] == 2) & (train_set['Mitoses'] == 4)].shape[0]],
        [train_set[(train_set['class'] == 2) & (train_set['Uniformity of Clump Thickness'] == 5)].shape[0],train_set[(train_set['class'] == 2) & (train_set['Uniformity of Cell Size'] == 5)].shape[0],train_set[(train_set['class'] == 2) & (train_set['Cell Shape'] == 5)].shape[0],train_set[(train_set['class'] == 2) & (train_set['Marginal Adhesion'] == 5)].shape[0],train_set[(train_set['class'] == 2) & (train_set['Single Epithelial Cell Size'] == 5)].shape[0],train_set[(train_set['class'] == 2) & (train_set['Bare Nuclei'] == '5')].shape[0],train_set[(train_set['class'] == 2) & (train_set['Bland Chromatin'] == 5)].shape[0],train_set[(train_set['class'] == 2) & (train_set['Normal Nucleoli'] == 5)].shape[0],train_set[(train_set['class'] == 2) & (train_set['Mitoses'] == 5)].shape[0]],
        [train_set[(train_set['class'] == 2) & (train_set['Uniformity of Clump Thickness'] == 6)].shape[0],train_set[(train_set['class'] == 2) & (train_set['Uniformity of Cell Size'] == 6)].shape[0],train_set[(train_set['class'] == 2) & (train_set['Cell Shape'] == 6)].shape[0],train_set[(train_set['class'] == 2) & (train_set['Marginal Adhesion'] == 6)].shape[0],train_set[(train_set['class'] == 2) & (train_set['Single Epithelial Cell Size'] == 6)].shape[0],train_set[(train_set['class'] == 2) & (train_set['Bare Nuclei'] == '6')].shape[0],train_set[(train_set['class'] == 2) & (train_set['Bland Chromatin'] == 6)].shape[0],train_set[(train_set['class'] == 2) & (train_set['Normal Nucleoli'] == 6)].shape[0],train_set[(train_set['class'] == 2) & (train_set['Mitoses'] == 6)].shape[0]],
        [train_set[(train_set['class'] == 2) & (train_set['Uniformity of Clump Thickness'] == 7)].shape[0],train_set[(train_set['class'] == 2) & (train_set['Uniformity of Cell Size'] == 7)].shape[0],train_set[(train_set['class'] == 2) & (train_set['Cell Shape'] == 7)].shape[0],train_set[(train_set['class'] == 2) & (train_set['Marginal Adhesion'] == 7)].shape[0],train_set[(train_set['class'] == 2) & (train_set['Single Epithelial Cell Size'] == 7)].shape[0],train_set[(train_set['class'] == 2) & (train_set['Bare Nuclei'] == '7')].shape[0],train_set[(train_set['class'] == 2) & (train_set['Bland Chromatin'] == 7)].shape[0],train_set[(train_set['class'] == 2) & (train_set['Normal Nucleoli'] == 7)].shape[0],train_set[(train_set['class'] == 2) & (train_set['Mitoses'] == 7)].shape[0]],
        [train_set[(train_set['class'] == 2) & (train_set['Uniformity of Clump Thickness'] == 8)].shape[0],train_set[(train_set['class'] == 2) & (train_set['Uniformity of Cell Size'] == 8)].shape[0],train_set[(train_set['class'] == 2) & (train_set['Cell Shape'] == 8)].shape[0],train_set[(train_set['class'] == 2) & (train_set['Marginal Adhesion'] == 8)].shape[0],train_set[(train_set['class'] == 2) & (train_set['Single Epithelial Cell Size'] == 8)].shape[0],train_set[(train_set['class'] == 2) & (train_set['Bare Nuclei'] == '8')].shape[0],train_set[(train_set['class'] == 2) & (train_set['Bland Chromatin'] == 8)].shape[0],train_set[(train_set['class'] == 2) & (train_set['Normal Nucleoli'] == 8)].shape[0],train_set[(train_set['class'] == 2) & (train_set['Mitoses'] == 8)].shape[0]],
        [train_set[(train_set['class'] == 2) & (train_set['Uniformity of Clump Thickness'] == 9)].shape[0],train_set[(train_set['class'] == 2) & (train_set['Uniformity of Cell Size'] == 9)].shape[0],train_set[(train_set['class'] == 2) & (train_set['Cell Shape'] == 9)].shape[0],train_set[(train_set['class'] == 2) & (train_set['Marginal Adhesion'] == 9)].shape[0],train_set[(train_set['class'] == 2) & (train_set['Single Epithelial Cell Size'] == 9)].shape[0],train_set[(train_set['class'] == 2) & (train_set['Bare Nuclei'] == '9')].shape[0],train_set[(train_set['class'] == 2) & (train_set['Bland Chromatin'] == 9)].shape[0],train_set[(train_set['class'] == 2) & (train_set['Normal Nucleoli'] == 9)].shape[0],train_set[(train_set['class'] == 2) & (train_set['Mitoses'] == 9)].shape[0]],
        [train_set[(train_set['class'] == 2) & (train_set['Uniformity of Clump Thickness'] == 10)].shape[0],train_set[(train_set['class'] == 2) & (train_set['Uniformity of Cell Size'] == 10)].shape[0],train_set[(train_set['class'] == 2) & (train_set['Cell Shape'] == 10)].shape[0],train_set[(train_set['class'] == 2) & (train_set['Marginal Adhesion'] == 10)].shape[0],train_set[(train_set['class'] == 2) & (train_set['Single Epithelial Cell Size'] == 10)].shape[0],train_set[(train_set['class'] == 2) & (train_set['Bare Nuclei'] == '10')].shape[0],train_set[(train_set['class'] == 2) & (train_set['Bland Chromatin'] == 10)].shape[0],train_set[(train_set['class'] == 2) & (train_set['Normal Nucleoli'] == 10)].shape[0],train_set[(train_set['class'] == 2) & (train_set['Mitoses'] == 10)].shape[0]],
        [train_set[(train_set['class'] == 4) & (train_set['Uniformity of Clump Thickness'] == 1)].shape[0],train_set[(train_set['class'] == 4) & (train_set['Uniformity of Cell Size'] == 1)].shape[0],train_set[(train_set['class'] == 4) & (train_set['Cell Shape'] == 1)].shape[0],train_set[(train_set['class'] == 4) & (train_set['Marginal Adhesion'] == 1)].shape[0],train_set[(train_set['class'] == 4) & (train_set['Single Epithelial Cell Size'] == 1)].shape[0],train_set[(train_set['class'] == 4) & (train_set['Bare Nuclei'] == '1')].shape[0],train_set[(train_set['class'] == 4) & (train_set['Bland Chromatin'] == 1)].shape[0],train_set[(train_set['class'] == 4) & (train_set['Normal Nucleoli'] == 1)].shape[0],train_set[(train_set['class'] == 4) & (train_set['Mitoses'] == 1)].shape[0]],
        [train_set[(train_set['class'] == 4) & (train_set['Uniformity of Clump Thickness'] == 2)].shape[0],train_set[(train_set['class'] == 4) & (train_set['Uniformity of Cell Size'] == 2)].shape[0],train_set[(train_set['class'] == 4) & (train_set['Cell Shape'] == 2)].shape[0],train_set[(train_set['class'] == 4) & (train_set['Marginal Adhesion'] == 2)].shape[0],train_set[(train_set['class'] == 4) & (train_set['Single Epithelial Cell Size'] == 2)].shape[0],train_set[(train_set['class'] == 4) & (train_set['Bare Nuclei'] == '2')].shape[0],train_set[(train_set['class'] == 4) & (train_set['Bland Chromatin'] == 2)].shape[0],train_set[(train_set['class'] == 4) & (train_set['Normal Nucleoli'] == 2)].shape[0],train_set[(train_set['class'] == 4) & (train_set['Mitoses'] == 2)].shape[0]],
        [train_set[(train_set['class'] == 4) & (train_set['Uniformity of Clump Thickness'] == 3)].shape[0],train_set[(train_set['class'] == 4) & (train_set['Uniformity of Cell Size'] == 3)].shape[0],train_set[(train_set['class'] == 4) & (train_set['Cell Shape'] == 3)].shape[0],train_set[(train_set['class'] == 4) & (train_set['Marginal Adhesion'] == 3)].shape[0],train_set[(train_set['class'] == 4) & (train_set['Single Epithelial Cell Size'] == 3)].shape[0],train_set[(train_set['class'] == 4) & (train_set['Bare Nuclei'] == '3')].shape[0],train_set[(train_set['class'] == 4) & (train_set['Bland Chromatin'] == 3)].shape[0],train_set[(train_set['class'] == 4) & (train_set['Normal Nucleoli'] == 3)].shape[0],train_set[(train_set['class'] == 4) & (train_set['Mitoses'] == 3)].shape[0]],
        [train_set[(train_set['class'] == 4) & (train_set['Uniformity of Clump Thickness'] == 4)].shape[0],train_set[(train_set['class'] == 4) & (train_set['Uniformity of Cell Size'] == 4)].shape[0],train_set[(train_set['class'] == 4) & (train_set['Cell Shape'] == 4)].shape[0],train_set[(train_set['class'] == 4) & (train_set['Marginal Adhesion'] == 4)].shape[0],train_set[(train_set['class'] == 4) & (train_set['Single Epithelial Cell Size'] == 4)].shape[0],train_set[(train_set['class'] == 4) & (train_set['Bare Nuclei'] == '4')].shape[0],train_set[(train_set['class'] == 4) & (train_set['Bland Chromatin'] == 4)].shape[0],train_set[(train_set['class'] == 4) & (train_set['Normal Nucleoli'] == 4)].shape[0],train_set[(train_set['class'] == 4) & (train_set['Mitoses'] == 4)].shape[0]],
        [train_set[(train_set['class'] == 4) & (train_set['Uniformity of Clump Thickness'] == 5)].shape[0],train_set[(train_set['class'] == 4) & (train_set['Uniformity of Cell Size'] == 5)].shape[0],train_set[(train_set['class'] == 4) & (train_set['Cell Shape'] == 5)].shape[0],train_set[(train_set['class'] == 4) & (train_set['Marginal Adhesion'] == 5)].shape[0],train_set[(train_set['class'] == 4) & (train_set['Single Epithelial Cell Size'] == 5)].shape[0],train_set[(train_set['class'] == 4) & (train_set['Bare Nuclei'] == '5')].shape[0],train_set[(train_set['class'] == 4) & (train_set['Bland Chromatin'] == 5)].shape[0],train_set[(train_set['class'] == 4) & (train_set['Normal Nucleoli'] == 5)].shape[0],train_set[(train_set['class'] == 4) & (train_set['Mitoses'] == 5)].shape[0]],
        [train_set[(train_set['class'] == 4) & (train_set['Uniformity of Clump Thickness'] == 6)].shape[0],train_set[(train_set['class'] == 4) & (train_set['Uniformity of Cell Size'] == 6)].shape[0],train_set[(train_set['class'] == 4) & (train_set['Cell Shape'] == 6)].shape[0],train_set[(train_set['class'] == 4) & (train_set['Marginal Adhesion'] == 6)].shape[0],train_set[(train_set['class'] == 4) & (train_set['Single Epithelial Cell Size'] == 6)].shape[0],train_set[(train_set['class'] == 4) & (train_set['Bare Nuclei'] == '6')].shape[0],train_set[(train_set['class'] == 4) & (train_set['Bland Chromatin'] == 6)].shape[0],train_set[(train_set['class'] == 4) & (train_set['Normal Nucleoli'] == 6)].shape[0],train_set[(train_set['class'] == 4) & (train_set['Mitoses'] == 6)].shape[0]],
        [train_set[(train_set['class'] == 4) & (train_set['Uniformity of Clump Thickness'] == 7)].shape[0],train_set[(train_set['class'] == 4) & (train_set['Uniformity of Cell Size'] == 7)].shape[0],train_set[(train_set['class'] == 4) & (train_set['Cell Shape'] == 7)].shape[0],train_set[(train_set['class'] == 4) & (train_set['Marginal Adhesion'] == 7)].shape[0],train_set[(train_set['class'] == 4) & (train_set['Single Epithelial Cell Size'] == 7)].shape[0],train_set[(train_set['class'] == 4) & (train_set['Bare Nuclei'] == '7')].shape[0],train_set[(train_set['class'] == 4) & (train_set['Bland Chromatin'] == 7)].shape[0],train_set[(train_set['class'] == 4) & (train_set['Normal Nucleoli'] == 7)].shape[0],train_set[(train_set['class'] == 4) & (train_set['Mitoses'] == 7)].shape[0]],
        [train_set[(train_set['class'] == 4) & (train_set['Uniformity of Clump Thickness'] == 8)].shape[0],train_set[(train_set['class'] == 4) & (train_set['Uniformity of Cell Size'] == 8)].shape[0],train_set[(train_set['class'] == 4) & (train_set['Cell Shape'] == 8)].shape[0],train_set[(train_set['class'] == 4) & (train_set['Marginal Adhesion'] == 8)].shape[0],train_set[(train_set['class'] == 4) & (train_set['Single Epithelial Cell Size'] == 8)].shape[0],train_set[(train_set['class'] == 4) & (train_set['Bare Nuclei'] == '8')].shape[0],train_set[(train_set['class'] == 4) & (train_set['Bland Chromatin'] == 8)].shape[0],train_set[(train_set['class'] == 4) & (train_set['Normal Nucleoli'] == 8)].shape[0],train_set[(train_set['class'] == 4) & (train_set['Mitoses'] == 8)].shape[0]],
        [train_set[(train_set['class'] == 4) & (train_set['Uniformity of Clump Thickness'] == 9)].shape[0],train_set[(train_set['class'] == 4) & (train_set['Uniformity of Cell Size'] == 9)].shape[0],train_set[(train_set['class'] == 4) & (train_set['Cell Shape'] == 9)].shape[0],train_set[(train_set['class'] == 4) & (train_set['Marginal Adhesion'] == 9)].shape[0],train_set[(train_set['class'] == 4) & (train_set['Single Epithelial Cell Size'] == 9)].shape[0],train_set[(train_set['class'] == 4) & (train_set['Bare Nuclei'] == '9')].shape[0],train_set[(train_set['class'] == 4) & (train_set['Bland Chromatin'] == 9)].shape[0],train_set[(train_set['class'] == 4) & (train_set['Normal Nucleoli'] == 9)].shape[0],train_set[(train_set['class'] == 4) & (train_set['Mitoses'] == 9)].shape[0]],
        [train_set[(train_set['class'] == 4) & (train_set['Uniformity of Clump Thickness'] == 10)].shape[0],train_set[(train_set['class'] == 4) & (train_set['Uniformity of Cell Size'] == 10)].shape[0],train_set[(train_set['class'] == 4) & (train_set['Cell Shape'] == 10)].shape[0],train_set[(train_set['class'] == 4) & (train_set['Marginal Adhesion'] == 10)].shape[0],train_set[(train_set['class'] == 4) & (train_set['Single Epithelial Cell Size'] == 10)].shape[0],train_set[(train_set['class'] == 4) & (train_set['Bare Nuclei'] == '10')].shape[0],train_set[(train_set['class'] == 4) & (train_set['Bland Chromatin'] == 10)].shape[0],train_set[(train_set['class'] == 4) & (train_set['Normal Nucleoli'] == 10)].shape[0],train_set[(train_set['class'] == 4) & (train_set['Mitoses'] == 10)].shape[0]]
    ]

    train_set_frequency_table = pd.DataFrame(data_struct, columns=["Uniformity of Clump Thickness", "Uniformity of Cell Size", "Cell Shape", "Marginal Adhesion", "Single Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses"], index=['1-2','2-2','3-2','4-2','5-2','6-2','7-2','8-2','9-2','10-2','1-4','2-4','3-4','4-4','5-4','6-4','7-4','8-4','9-4','10-4'])

    return train_set_frequency_table

# create frequency tables for a given votes data segment
def votes_frequency(train_set):
    data_struct = [
    [train_set[(train_set['class'] == 'republican') & (train_set['infants'] == 'y')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['water'] == 'y')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['adoption'] == 'y')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['physician'] == 'y')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['salvador'] == 'y')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['religious'] == 'y')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['satellite'] == 'y')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['nicaragua'] == 'y')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['missile'] == 'y')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['immigration'] == 'y')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['synfuels'] == 'y')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['education'] == 'y')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['superfund'] == 'y')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['crime'] == 'y')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['exports'] == 'y')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['south-africa'] == 'y')].shape[0]],
    [train_set[(train_set['class'] == 'republican') & (train_set['infants'] == 'n')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['water'] == 'n')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['adoption'] == 'n')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['physician'] == 'n')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['salvador'] == 'n')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['religious'] == 'n')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['satellite'] == 'n')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['nicaragua'] == 'n')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['missile'] == 'n')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['immigration'] == 'n')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['synfuels'] == 'n')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['education'] == 'n')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['superfund'] == 'n')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['crime'] == 'n')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['exports'] == 'n')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['south-africa'] == 'n')].shape[0]],
    [train_set[(train_set['class'] == 'republican') & (train_set['infants'] == '?')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['water'] == '?')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['adoption'] == '?')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['physician'] == '?')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['salvador'] == '?')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['religious'] == '?')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['satellite'] == '?')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['nicaragua'] == '?')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['missile'] == '?')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['immigration'] == '?')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['synfuels'] == '?')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['education'] == '?')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['superfund'] == '?')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['crime'] == '?')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['exports'] == '?')].shape[0],train_set[(train_set['class'] == 'republican') & (train_set['south-africa'] == '?')].shape[0]],
    [train_set[(train_set['class'] == 'democrat') & (train_set['infants'] == 'y')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['water'] == 'y')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['adoption'] == 'y')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['physician'] == 'y')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['salvador'] == 'y')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['religious'] == 'y')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['satellite'] == 'y')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['nicaragua'] == 'y')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['missile'] == 'y')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['immigration'] == 'y')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['synfuels'] == 'y')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['education'] == 'y')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['superfund'] == 'y')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['crime'] == 'y')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['exports'] == 'y')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['south-africa'] == 0)].shape[0]],
    [train_set[(train_set['class'] == 'democrat') & (train_set['infants'] == 1)].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['water'] == 'n')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['adoption'] == 'n')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['physician'] == 'n')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['salvador'] == 'n')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['religious'] == 'n')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['satellite'] == 'n')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['nicaragua'] == 'n')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['missile'] == 'n')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['immigration'] == 'n')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['synfuels'] == 'n')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['education'] == 'n')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['superfund'] == 'n')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['crime'] == 'n')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['exports'] == 'n')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['south-africa'] == 'n')].shape[0]],
    [train_set[(train_set['class'] == 'democrat') & (train_set['infants'] == '?')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['water'] == '?')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['adoption'] == '?')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['physician'] == '?')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['salvador'] == '?')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['religious'] == '?')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['satellite'] == '?')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['nicaragua'] == '?')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['missile'] == '?')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['immigration'] == '?')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['synfuels'] == '?')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['education'] == '?')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['superfund'] == '?')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['crime'] == '?')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['exports'] == '?')].shape[0],train_set[(train_set['class'] == 'democrat') & (train_set['south-africa'] == '?')].shape[0]]
]
    vote_set_frequency_table = pd.DataFrame(data_struct, columns=["infants","water","adoption","physician","salvador","religious","satellite","nicaragua","missile","immigration","synfuels","education","superfund","crime","exports","south-africa"], index=['y-republican','n-republican','?-republican','y-democrat','n-democrat','?-democrat'])

    return vote_set_frequency_table

# create frequency tables for a given soybeans data segment
def soy_frequency(train_set):
    data_struct = [
    [train_set[(train_set['class'] == 'D1') & (train_set['date'] == 0)].shape[0],train_set[(train_set['class'] == 'D1') & (train_set['plant-stand'] == 0)].shape[0], train_set[(train_set['class'] == 'D1') & (train_set['precip'] == 0)].shape[0],train_set[(train_set['class'] == 'D1') & (train_set['temp'] == 0)].shape[0], train_set[(train_set['class'] == 'D1') & (train_set['hail'] == 0)].shape[0],
    train_set[(train_set['class'] == 'D1') & (train_set['crop-hist'] == 0)].shape[0],train_set[(train_set['class'] == 'D1') & (train_set['area-damaged'] == 0)].shape[0], train_set[(train_set['class'] == 'D1') & (train_set['severity'] == 0)].shape[0],train_set[(train_set['class'] == 'D1') & (train_set['seed-tmt'] == 0)].shape[0], train_set[(train_set['class'] == 'D1') & (train_set['germination'] == 0)].shape[0],train_set[(train_set['class'] == 'D1') & (train_set['leaves'] == 0)].shape[0], train_set[(train_set['class'] == 'D1') & (train_set['lodging'] == 0)].shape[0],
    train_set[(train_set['class'] == 'D1') & (train_set['stem-cankers'] == 0)].shape[0],train_set[(train_set['class'] == 'D1') & (train_set['canker-lesion'] == 0)].shape[0], train_set[(train_set['class'] == 'D1') & (train_set['fruiting-bodies'] == 0)].shape[0],train_set[(train_set['class'] == 'D1') & (train_set['external-decay'] == 0)].shape[0], train_set[(train_set['class'] == 'D1') & (train_set['mycelium'] == 0)].shape[0],
    train_set[(train_set['class'] == 'D1') & (train_set['int-discolor'] == 0)].shape[0],train_set[(train_set['class'] == 'D1') & (train_set['sclerotia'] == 0)].shape[0], train_set[(train_set['class'] == 'D1') & (train_set['fruit-pods'] == 0)].shape[0], train_set[(train_set['class'] == 'D1') & (train_set['roots'] == 0)].shape[0]],

    [train_set[(train_set['class'] == 'D1') & (train_set['date'] == 1)].shape[0],train_set[(train_set['class'] == 'D1') & (train_set['plant-stand'] == 1)].shape[0], train_set[(train_set['class'] == 'D1') & (train_set['precip'] == 1)].shape[0],train_set[(train_set['class'] == 'D1') & (train_set['temp'] == 1)].shape[0], train_set[(train_set['class'] == 'D1') & (train_set['hail'] == 1)].shape[0],
    train_set[(train_set['class'] == 'D1') & (train_set['crop-hist'] == 1)].shape[0],train_set[(train_set['class'] == 'D1') & (train_set['area-damaged'] == 1)].shape[0], train_set[(train_set['class'] == 'D1') & (train_set['severity'] == 1)].shape[0],train_set[(train_set['class'] == 'D1') & (train_set['seed-tmt'] == 1)].shape[0], train_set[(train_set['class'] == 'D1') & (train_set['germination'] == 1)].shape[0],train_set[(train_set['class'] == 'D1') & (train_set['leaves'] == 1)].shape[0], train_set[(train_set['class'] == 'D1') & (train_set['lodging'] == 1)].shape[0],
    train_set[(train_set['class'] == 'D1') & (train_set['stem-cankers'] == 1)].shape[0],train_set[(train_set['class'] == 'D1') & (train_set['canker-lesion'] == 1)].shape[0], train_set[(train_set['class'] == 'D1') & (train_set['fruiting-bodies'] == 1)].shape[0],train_set[(train_set['class'] == 'D1') & (train_set['external-decay'] == 1)].shape[0], train_set[(train_set['class'] == 'D1') & (train_set['mycelium'] == 1)].shape[0],
    train_set[(train_set['class'] == 'D1') & (train_set['int-discolor'] == 1)].shape[0],train_set[(train_set['class'] == 'D1') & (train_set['sclerotia'] == 1)].shape[0], train_set[(train_set['class'] == 'D1') & (train_set['fruit-pods'] == 1)].shape[0], train_set[(train_set['class'] == 'D1') & (train_set['roots'] == 1)].shape[0]],

    [train_set[(train_set['class'] == 'D1') & (train_set['date'] == 2)].shape[0],train_set[(train_set['class'] == 'D1') & (train_set['plant-stand'] == 2)].shape[0], train_set[(train_set['class'] == 'D1') & (train_set['precip'] == 2)].shape[0],train_set[(train_set['class'] == 'D1') & (train_set['temp'] == 2)].shape[0], train_set[(train_set['class'] == 'D1') & (train_set['hail'] == 2)].shape[0],
    train_set[(train_set['class'] == 'D1') & (train_set['crop-hist'] == 2)].shape[0],train_set[(train_set['class'] == 'D1') & (train_set['area-damaged'] == 2)].shape[0], train_set[(train_set['class'] == 'D1') & (train_set['severity'] == 2)].shape[0],train_set[(train_set['class'] == 'D1') & (train_set['seed-tmt'] == 2)].shape[0], train_set[(train_set['class'] == 'D1') & (train_set['germination'] == 2)].shape[0],train_set[(train_set['class'] == 'D1') & (train_set['leaves'] == 2)].shape[0], train_set[(train_set['class'] == 'D1') & (train_set['lodging'] == 2)].shape[0],
    train_set[(train_set['class'] == 'D1') & (train_set['stem-cankers'] == 2)].shape[0],train_set[(train_set['class'] == 'D1') & (train_set['canker-lesion'] == 2)].shape[0], train_set[(train_set['class'] == 'D1') & (train_set['fruiting-bodies'] == 2)].shape[0],train_set[(train_set['class'] == 'D1') & (train_set['external-decay'] == 2)].shape[0], train_set[(train_set['class'] == 'D1') & (train_set['mycelium'] == 2)].shape[0],
    train_set[(train_set['class'] == 'D1') & (train_set['int-discolor'] == 2)].shape[0],train_set[(train_set['class'] == 'D1') & (train_set['sclerotia'] == 2)].shape[0], train_set[(train_set['class'] == 'D1') & (train_set['fruit-pods'] == 2)].shape[0], train_set[(train_set['class'] == 'D1') & (train_set['roots'] == 2)].shape[0]],

    [train_set[(train_set['class'] == 'D1') & (train_set['date'] == 3)].shape[0],train_set[(train_set['class'] == 'D1') & (train_set['plant-stand'] == 3)].shape[0], train_set[(train_set['class'] == 'D1') & (train_set['precip'] == 3)].shape[0],train_set[(train_set['class'] == 'D1') & (train_set['temp'] == 3)].shape[0], train_set[(train_set['class'] == 'D1') & (train_set['hail'] == 3)].shape[0],
    train_set[(train_set['class'] == 'D1') & (train_set['crop-hist'] == 3)].shape[0],train_set[(train_set['class'] == 'D1') & (train_set['area-damaged'] == 3)].shape[0], train_set[(train_set['class'] == 'D1') & (train_set['severity'] == 3)].shape[0],train_set[(train_set['class'] == 'D1') & (train_set['seed-tmt'] == 3)].shape[0], train_set[(train_set['class'] == 'D1') & (train_set['germination'] == 3)].shape[0],train_set[(train_set['class'] == 'D1') & (train_set['leaves'] == 3)].shape[0], train_set[(train_set['class'] == 'D1') & (train_set['lodging'] == 3)].shape[0],
    train_set[(train_set['class'] == 'D1') & (train_set['stem-cankers'] == 3)].shape[0],train_set[(train_set['class'] == 'D1') & (train_set['canker-lesion'] == 3)].shape[0], train_set[(train_set['class'] == 'D1') & (train_set['fruiting-bodies'] == 3)].shape[0],train_set[(train_set['class'] == 'D1') & (train_set['external-decay'] == 3)].shape[0], train_set[(train_set['class'] == 'D1') & (train_set['mycelium'] == 3)].shape[0],
    train_set[(train_set['class'] == 'D1') & (train_set['int-discolor'] == 3)].shape[0],train_set[(train_set['class'] == 'D1') & (train_set['sclerotia'] == 3)].shape[0], train_set[(train_set['class'] == 'D1') & (train_set['fruit-pods'] == 3)].shape[0], train_set[(train_set['class'] == 'D1') & (train_set['roots'] == 3)].shape[0]],

    [train_set[(train_set['class'] == 'D1') & (train_set['date'] == 4)].shape[0],train_set[(train_set['class'] == 'D1') & (train_set['plant-stand'] == 4)].shape[0], train_set[(train_set['class'] == 'D1') & (train_set['precip'] == 4)].shape[0],train_set[(train_set['class'] == 'D1') & (train_set['temp'] == 4)].shape[0], train_set[(train_set['class'] == 'D1') & (train_set['hail'] == 4)].shape[0],
    train_set[(train_set['class'] == 'D1') & (train_set['crop-hist'] == 4)].shape[0],train_set[(train_set['class'] == 'D1') & (train_set['area-damaged'] == 4)].shape[0], train_set[(train_set['class'] == 'D1') & (train_set['severity'] == 4)].shape[0],train_set[(train_set['class'] == 'D1') & (train_set['seed-tmt'] == 4)].shape[0], train_set[(train_set['class'] == 'D1') & (train_set['germination'] == 4)].shape[0],train_set[(train_set['class'] == 'D1') & (train_set['leaves'] == 4)].shape[0], train_set[(train_set['class'] == 'D1') & (train_set['lodging'] == 4)].shape[0],
    train_set[(train_set['class'] == 'D1') & (train_set['stem-cankers'] == 4)].shape[0],train_set[(train_set['class'] == 'D1') & (train_set['canker-lesion'] == 4)].shape[0], train_set[(train_set['class'] == 'D1') & (train_set['fruiting-bodies'] == 4)].shape[0],train_set[(train_set['class'] == 'D1') & (train_set['external-decay'] == 4)].shape[0], train_set[(train_set['class'] == 'D1') & (train_set['mycelium'] == 4)].shape[0],
    train_set[(train_set['class'] == 'D1') & (train_set['int-discolor'] == 4)].shape[0],train_set[(train_set['class'] == 'D1') & (train_set['sclerotia'] == 4)].shape[0], train_set[(train_set['class'] == 'D1') & (train_set['fruit-pods'] == 4)].shape[0], train_set[(train_set['class'] == 'D1') & (train_set['roots'] == 4)].shape[0]],

    [train_set[(train_set['class'] == 'D1') & (train_set['date'] == 5)].shape[0],train_set[(train_set['class'] == 'D1') & (train_set['plant-stand'] == 5)].shape[0], train_set[(train_set['class'] == 'D1') & (train_set['precip'] == 5)].shape[0],train_set[(train_set['class'] == 'D1') & (train_set['temp'] == 5)].shape[0], train_set[(train_set['class'] == 'D1') & (train_set['hail'] == 5)].shape[0],
    train_set[(train_set['class'] == 'D1') & (train_set['crop-hist'] == 5)].shape[0],train_set[(train_set['class'] == 'D1') & (train_set['area-damaged'] == 5)].shape[0], train_set[(train_set['class'] == 'D1') & (train_set['severity'] == 5)].shape[0],train_set[(train_set['class'] == 'D1') & (train_set['seed-tmt'] == 5)].shape[0], train_set[(train_set['class'] == 'D1') & (train_set['germination'] == 5)].shape[0],train_set[(train_set['class'] == 'D1') & (train_set['leaves'] == 5)].shape[0], train_set[(train_set['class'] == 'D1') & (train_set['lodging'] == 5)].shape[0],
    train_set[(train_set['class'] == 'D1') & (train_set['stem-cankers'] == 5)].shape[0],train_set[(train_set['class'] == 'D1') & (train_set['canker-lesion'] == 5)].shape[0], train_set[(train_set['class'] == 'D1') & (train_set['fruiting-bodies'] == 5)].shape[0],train_set[(train_set['class'] == 'D1') & (train_set['external-decay'] == 5)].shape[0], train_set[(train_set['class'] == 'D1') & (train_set['mycelium'] == 5)].shape[0],
    train_set[(train_set['class'] == 'D1') & (train_set['int-discolor'] == 5)].shape[0],train_set[(train_set['class'] == 'D1') & (train_set['sclerotia'] == 5)].shape[0], train_set[(train_set['class'] == 'D1') & (train_set['fruit-pods'] == 5)].shape[0], train_set[(train_set['class'] == 'D1') & (train_set['roots'] == 5)].shape[0]],

    [train_set[(train_set['class'] == 'D1') & (train_set['date'] == 6)].shape[0],train_set[(train_set['class'] == 'D1') & (train_set['plant-stand'] == 6)].shape[0], train_set[(train_set['class'] == 'D1') & (train_set['precip'] == 6)].shape[0],train_set[(train_set['class'] == 'D1') & (train_set['temp'] == 6)].shape[0], train_set[(train_set['class'] == 'D1') & (train_set['hail'] == 6)].shape[0],
    train_set[(train_set['class'] == 'D1') & (train_set['crop-hist'] == 6)].shape[0],train_set[(train_set['class'] == 'D1') & (train_set['area-damaged'] == 6)].shape[0], train_set[(train_set['class'] == 'D1') & (train_set['severity'] == 6)].shape[0],train_set[(train_set['class'] == 'D1') & (train_set['seed-tmt'] == 6)].shape[0], train_set[(train_set['class'] == 'D1') & (train_set['germination'] == 6)].shape[0],train_set[(train_set['class'] == 'D1') & (train_set['leaves'] == 6)].shape[0], train_set[(train_set['class'] == 'D1') & (train_set['lodging'] == 6)].shape[0],
    train_set[(train_set['class'] == 'D1') & (train_set['stem-cankers'] == 6)].shape[0],train_set[(train_set['class'] == 'D1') & (train_set['canker-lesion'] == 6)].shape[0], train_set[(train_set['class'] == 'D1') & (train_set['fruiting-bodies'] == 6)].shape[0],train_set[(train_set['class'] == 'D1') & (train_set['external-decay'] == 6)].shape[0], train_set[(train_set['class'] == 'D1') & (train_set['mycelium'] == 6)].shape[0],
    train_set[(train_set['class'] == 'D1') & (train_set['int-discolor'] == 6)].shape[0],train_set[(train_set['class'] == 'D1') & (train_set['sclerotia'] == 6)].shape[0], train_set[(train_set['class'] == 'D1') & (train_set['fruit-pods'] == 6)].shape[0], train_set[(train_set['class'] == 'D1') & (train_set['roots'] == 6)].shape[0]],

    [train_set[(train_set['class'] == 'D2') & (train_set['date'] == 0)].shape[0],train_set[(train_set['class'] == 'D2') & (train_set['plant-stand'] == 0)].shape[0], train_set[(train_set['class'] == 'D2') & (train_set['precip'] == 0)].shape[0],train_set[(train_set['class'] == 'D2') & (train_set['temp'] == 0)].shape[0], train_set[(train_set['class'] == 'D2') & (train_set['hail'] == 0)].shape[0],
    train_set[(train_set['class'] == 'D2') & (train_set['crop-hist'] == 0)].shape[0],train_set[(train_set['class'] == 'D2') & (train_set['area-damaged'] == 0)].shape[0], train_set[(train_set['class'] == 'D2') & (train_set['severity'] == 0)].shape[0],train_set[(train_set['class'] == 'D2') & (train_set['seed-tmt'] == 0)].shape[0], train_set[(train_set['class'] == 'D2') & (train_set['germination'] == 0)].shape[0],train_set[(train_set['class'] == 'D2') & (train_set['leaves'] == 0)].shape[0], train_set[(train_set['class'] == 'D2') & (train_set['lodging'] == 0)].shape[0],
    train_set[(train_set['class'] == 'D2') & (train_set['stem-cankers'] == 0)].shape[0],train_set[(train_set['class'] == 'D2') & (train_set['canker-lesion'] == 0)].shape[0], train_set[(train_set['class'] == 'D2') & (train_set['fruiting-bodies'] == 0)].shape[0],train_set[(train_set['class'] == 'D2') & (train_set['external-decay'] == 0)].shape[0], train_set[(train_set['class'] == 'D2') & (train_set['mycelium'] == 0)].shape[0],
    train_set[(train_set['class'] == 'D2') & (train_set['int-discolor'] == 0)].shape[0],train_set[(train_set['class'] == 'D2') & (train_set['sclerotia'] == 0)].shape[0], train_set[(train_set['class'] == 'D2') & (train_set['fruit-pods'] == 0)].shape[0], train_set[(train_set['class'] == 'D2') & (train_set['roots'] == 0)].shape[0]],

    [train_set[(train_set['class'] == 'D2') & (train_set['date'] == 1)].shape[0],train_set[(train_set['class'] == 'D2') & (train_set['plant-stand'] == 1)].shape[0], train_set[(train_set['class'] == 'D2') & (train_set['precip'] == 1)].shape[0],train_set[(train_set['class'] == 'D2') & (train_set['temp'] == 1)].shape[0], train_set[(train_set['class'] == 'D2') & (train_set['hail'] == 1)].shape[0],
    train_set[(train_set['class'] == 'D2') & (train_set['crop-hist'] == 1)].shape[0],train_set[(train_set['class'] == 'D2') & (train_set['area-damaged'] == 1)].shape[0], train_set[(train_set['class'] == 'D2') & (train_set['severity'] == 1)].shape[0],train_set[(train_set['class'] == 'D2') & (train_set['seed-tmt'] == 1)].shape[0], train_set[(train_set['class'] == 'D2') & (train_set['germination'] == 1)].shape[0],train_set[(train_set['class'] == 'D2') & (train_set['leaves'] == 1)].shape[0], train_set[(train_set['class'] == 'D2') & (train_set['lodging'] == 1)].shape[0],
    train_set[(train_set['class'] == 'D2') & (train_set['stem-cankers'] == 1)].shape[0],train_set[(train_set['class'] == 'D2') & (train_set['canker-lesion'] == 1)].shape[0], train_set[(train_set['class'] == 'D2') & (train_set['fruiting-bodies'] == 1)].shape[0],train_set[(train_set['class'] == 'D2') & (train_set['external-decay'] == 1)].shape[0], train_set[(train_set['class'] == 'D2') & (train_set['mycelium'] == 1)].shape[0],
    train_set[(train_set['class'] == 'D2') & (train_set['int-discolor'] == 1)].shape[0],train_set[(train_set['class'] == 'D2') & (train_set['sclerotia'] == 1)].shape[0], train_set[(train_set['class'] == 'D2') & (train_set['fruit-pods'] == 1)].shape[0], train_set[(train_set['class'] == 'D2') & (train_set['roots'] == 1)].shape[0]],

    [train_set[(train_set['class'] == 'D2') & (train_set['date'] == 2)].shape[0],train_set[(train_set['class'] == 'D2') & (train_set['plant-stand'] == 2)].shape[0], train_set[(train_set['class'] == 'D2') & (train_set['precip'] == 2)].shape[0],train_set[(train_set['class'] == 'D2') & (train_set['temp'] == 2)].shape[0], train_set[(train_set['class'] == 'D2') & (train_set['hail'] == 2)].shape[0],
    train_set[(train_set['class'] == 'D2') & (train_set['crop-hist'] == 2)].shape[0],train_set[(train_set['class'] == 'D2') & (train_set['area-damaged'] == 2)].shape[0], train_set[(train_set['class'] == 'D2') & (train_set['severity'] == 2)].shape[0],train_set[(train_set['class'] == 'D2') & (train_set['seed-tmt'] == 2)].shape[0], train_set[(train_set['class'] == 'D2') & (train_set['germination'] == 2)].shape[0],train_set[(train_set['class'] == 'D2') & (train_set['leaves'] == 2)].shape[0], train_set[(train_set['class'] == 'D2') & (train_set['lodging'] == 2)].shape[0],
    train_set[(train_set['class'] == 'D2') & (train_set['stem-cankers'] == 2)].shape[0],train_set[(train_set['class'] == 'D2') & (train_set['canker-lesion'] == 2)].shape[0], train_set[(train_set['class'] == 'D2') & (train_set['fruiting-bodies'] == 2)].shape[0],train_set[(train_set['class'] == 'D2') & (train_set['external-decay'] == 2)].shape[0], train_set[(train_set['class'] == 'D2') & (train_set['mycelium'] == 2)].shape[0],
    train_set[(train_set['class'] == 'D2') & (train_set['int-discolor'] == 2)].shape[0],train_set[(train_set['class'] == 'D2') & (train_set['sclerotia'] == 2)].shape[0], train_set[(train_set['class'] == 'D2') & (train_set['fruit-pods'] == 2)].shape[0], train_set[(train_set['class'] == 'D2') & (train_set['roots'] == 2)].shape[0]],

    [train_set[(train_set['class'] == 'D2') & (train_set['date'] == 3)].shape[0],train_set[(train_set['class'] == 'D2') & (train_set['plant-stand'] == 3)].shape[0], train_set[(train_set['class'] == 'D2') & (train_set['precip'] == 3)].shape[0],train_set[(train_set['class'] == 'D2') & (train_set['temp'] == 3)].shape[0], train_set[(train_set['class'] == 'D2') & (train_set['hail'] == 3)].shape[0],
    train_set[(train_set['class'] == 'D2') & (train_set['crop-hist'] == 3)].shape[0],train_set[(train_set['class'] == 'D2') & (train_set['area-damaged'] == 3)].shape[0], train_set[(train_set['class'] == 'D2') & (train_set['severity'] == 3)].shape[0],train_set[(train_set['class'] == 'D2') & (train_set['seed-tmt'] == 3)].shape[0], train_set[(train_set['class'] == 'D2') & (train_set['germination'] == 3)].shape[0],train_set[(train_set['class'] == 'D2') & (train_set['leaves'] == 3)].shape[0], train_set[(train_set['class'] == 'D2') & (train_set['lodging'] == 3)].shape[0],
    train_set[(train_set['class'] == 'D2') & (train_set['stem-cankers'] == 3)].shape[0],train_set[(train_set['class'] == 'D2') & (train_set['canker-lesion'] == 3)].shape[0], train_set[(train_set['class'] == 'D2') & (train_set['fruiting-bodies'] == 3)].shape[0],train_set[(train_set['class'] == 'D2') & (train_set['external-decay'] == 3)].shape[0], train_set[(train_set['class'] == 'D2') & (train_set['mycelium'] == 3)].shape[0],
    train_set[(train_set['class'] == 'D2') & (train_set['int-discolor'] == 3)].shape[0],train_set[(train_set['class'] == 'D2') & (train_set['sclerotia'] == 3)].shape[0], train_set[(train_set['class'] == 'D2') & (train_set['fruit-pods'] == 3)].shape[0], train_set[(train_set['class'] == 'D2') & (train_set['roots'] == 3)].shape[0]],

    [train_set[(train_set['class'] == 'D2') & (train_set['date'] == 4)].shape[0],train_set[(train_set['class'] == 'D2') & (train_set['plant-stand'] == 4)].shape[0], train_set[(train_set['class'] == 'D2') & (train_set['precip'] == 4)].shape[0],train_set[(train_set['class'] == 'D2') & (train_set['temp'] == 4)].shape[0], train_set[(train_set['class'] == 'D2') & (train_set['hail'] == 4)].shape[0],
    train_set[(train_set['class'] == 'D2') & (train_set['crop-hist'] == 4)].shape[0],train_set[(train_set['class'] == 'D2') & (train_set['area-damaged'] == 4)].shape[0], train_set[(train_set['class'] == 'D2') & (train_set['severity'] == 4)].shape[0],train_set[(train_set['class'] == 'D2') & (train_set['seed-tmt'] == 4)].shape[0], train_set[(train_set['class'] == 'D2') & (train_set['germination'] == 4)].shape[0],train_set[(train_set['class'] == 'D2') & (train_set['leaves'] == 4)].shape[0], train_set[(train_set['class'] == 'D2') & (train_set['lodging'] == 4)].shape[0],
    train_set[(train_set['class'] == 'D2') & (train_set['stem-cankers'] == 4)].shape[0],train_set[(train_set['class'] == 'D2') & (train_set['canker-lesion'] == 4)].shape[0], train_set[(train_set['class'] == 'D2') & (train_set['fruiting-bodies'] == 4)].shape[0],train_set[(train_set['class'] == 'D2') & (train_set['external-decay'] == 4)].shape[0], train_set[(train_set['class'] == 'D2') & (train_set['mycelium'] == 4)].shape[0],
    train_set[(train_set['class'] == 'D2') & (train_set['int-discolor'] == 4)].shape[0],train_set[(train_set['class'] == 'D2') & (train_set['sclerotia'] == 4)].shape[0], train_set[(train_set['class'] == 'D2') & (train_set['fruit-pods'] == 4)].shape[0], train_set[(train_set['class'] == 'D2') & (train_set['roots'] == 4)].shape[0]],

    [train_set[(train_set['class'] == 'D2') & (train_set['date'] == 5)].shape[0],train_set[(train_set['class'] == 'D2') & (train_set['plant-stand'] == 5)].shape[0], train_set[(train_set['class'] == 'D2') & (train_set['precip'] == 5)].shape[0],train_set[(train_set['class'] == 'D2') & (train_set['temp'] == 5)].shape[0], train_set[(train_set['class'] == 'D2') & (train_set['hail'] == 5)].shape[0],
    train_set[(train_set['class'] == 'D2') & (train_set['crop-hist'] == 5)].shape[0],train_set[(train_set['class'] == 'D2') & (train_set['area-damaged'] == 5)].shape[0], train_set[(train_set['class'] == 'D2') & (train_set['severity'] == 5)].shape[0],train_set[(train_set['class'] == 'D2') & (train_set['seed-tmt'] == 5)].shape[0], train_set[(train_set['class'] == 'D2') & (train_set['germination'] == 5)].shape[0],train_set[(train_set['class'] == 'D2') & (train_set['leaves'] == 5)].shape[0], train_set[(train_set['class'] == 'D2') & (train_set['lodging'] == 5)].shape[0],
    train_set[(train_set['class'] == 'D2') & (train_set['stem-cankers'] == 5)].shape[0],train_set[(train_set['class'] == 'D2') & (train_set['canker-lesion'] == 5)].shape[0], train_set[(train_set['class'] == 'D2') & (train_set['fruiting-bodies'] == 5)].shape[0],train_set[(train_set['class'] == 'D2') & (train_set['external-decay'] == 5)].shape[0], train_set[(train_set['class'] == 'D2') & (train_set['mycelium'] == 5)].shape[0],
    train_set[(train_set['class'] == 'D2') & (train_set['int-discolor'] == 5)].shape[0],train_set[(train_set['class'] == 'D2') & (train_set['sclerotia'] == 5)].shape[0], train_set[(train_set['class'] == 'D2') & (train_set['fruit-pods'] == 5)].shape[0], train_set[(train_set['class'] == 'D2') & (train_set['roots'] == 5)].shape[0]],

    [train_set[(train_set['class'] == 'D2') & (train_set['date'] == 6)].shape[0],train_set[(train_set['class'] == 'D2') & (train_set['plant-stand'] == 6)].shape[0], train_set[(train_set['class'] == 'D2') & (train_set['precip'] == 6)].shape[0],train_set[(train_set['class'] == 'D2') & (train_set['temp'] == 6)].shape[0], train_set[(train_set['class'] == 'D2') & (train_set['hail'] == 6)].shape[0],
    train_set[(train_set['class'] == 'D2') & (train_set['crop-hist'] == 6)].shape[0],train_set[(train_set['class'] == 'D2') & (train_set['area-damaged'] == 6)].shape[0], train_set[(train_set['class'] == 'D2') & (train_set['severity'] == 6)].shape[0],train_set[(train_set['class'] == 'D2') & (train_set['seed-tmt'] == 6)].shape[0], train_set[(train_set['class'] == 'D2') & (train_set['germination'] == 6)].shape[0],train_set[(train_set['class'] == 'D2') & (train_set['leaves'] == 6)].shape[0], train_set[(train_set['class'] == 'D2') & (train_set['lodging'] == 6)].shape[0],
    train_set[(train_set['class'] == 'D2') & (train_set['stem-cankers'] == 6)].shape[0],train_set[(train_set['class'] == 'D2') & (train_set['canker-lesion'] == 6)].shape[0], train_set[(train_set['class'] == 'D2') & (train_set['fruiting-bodies'] == 6)].shape[0],train_set[(train_set['class'] == 'D2') & (train_set['external-decay'] == 6)].shape[0], train_set[(train_set['class'] == 'D2') & (train_set['mycelium'] == 6)].shape[0],
    train_set[(train_set['class'] == 'D2') & (train_set['int-discolor'] == 6)].shape[0],train_set[(train_set['class'] == 'D2') & (train_set['sclerotia'] == 6)].shape[0], train_set[(train_set['class'] == 'D2') & (train_set['fruit-pods'] == 6)].shape[0], train_set[(train_set['class'] == 'D2') & (train_set['roots'] == 6)].shape[0]],

    [train_set[(train_set['class'] == 'D3') & (train_set['date'] == 0)].shape[0],train_set[(train_set['class'] == 'D3') & (train_set['plant-stand'] == 0)].shape[0], train_set[(train_set['class'] == 'D3') & (train_set['precip'] == 0)].shape[0],train_set[(train_set['class'] == 'D3') & (train_set['temp'] == 0)].shape[0], train_set[(train_set['class'] == 'D3') & (train_set['hail'] == 0)].shape[0],
    train_set[(train_set['class'] == 'D3') & (train_set['crop-hist'] == 0)].shape[0],train_set[(train_set['class'] == 'D3') & (train_set['area-damaged'] == 0)].shape[0], train_set[(train_set['class'] == 'D3') & (train_set['severity'] == 0)].shape[0],train_set[(train_set['class'] == 'D3') & (train_set['seed-tmt'] == 0)].shape[0], train_set[(train_set['class'] == 'D3') & (train_set['germination'] == 0)].shape[0],train_set[(train_set['class'] == 'D3') & (train_set['leaves'] == 0)].shape[0], train_set[(train_set['class'] == 'D3') & (train_set['lodging'] == 0)].shape[0],
    train_set[(train_set['class'] == 'D3') & (train_set['stem-cankers'] == 0)].shape[0],train_set[(train_set['class'] == 'D3') & (train_set['canker-lesion'] == 0)].shape[0], train_set[(train_set['class'] == 'D3') & (train_set['fruiting-bodies'] == 0)].shape[0],train_set[(train_set['class'] == 'D3') & (train_set['external-decay'] == 0)].shape[0], train_set[(train_set['class'] == 'D3') & (train_set['mycelium'] == 0)].shape[0],
    train_set[(train_set['class'] == 'D3') & (train_set['int-discolor'] == 0)].shape[0],train_set[(train_set['class'] == 'D3') & (train_set['sclerotia'] == 0)].shape[0], train_set[(train_set['class'] == 'D3') & (train_set['fruit-pods'] == 0)].shape[0], train_set[(train_set['class'] == 'D3') & (train_set['roots'] == 0)].shape[0]],

    [train_set[(train_set['class'] == 'D3') & (train_set['date'] == 1)].shape[0],train_set[(train_set['class'] == 'D3') & (train_set['plant-stand'] == 1)].shape[0], train_set[(train_set['class'] == 'D3') & (train_set['precip'] == 1)].shape[0],train_set[(train_set['class'] == 'D3') & (train_set['temp'] == 1)].shape[0], train_set[(train_set['class'] == 'D3') & (train_set['hail'] == 1)].shape[0],
    train_set[(train_set['class'] == 'D3') & (train_set['crop-hist'] == 1)].shape[0],train_set[(train_set['class'] == 'D3') & (train_set['area-damaged'] == 1)].shape[0], train_set[(train_set['class'] == 'D3') & (train_set['severity'] == 1)].shape[0],train_set[(train_set['class'] == 'D3') & (train_set['seed-tmt'] == 1)].shape[0], train_set[(train_set['class'] == 'D3') & (train_set['germination'] == 1)].shape[0],train_set[(train_set['class'] == 'D3') & (train_set['leaves'] == 1)].shape[0], train_set[(train_set['class'] == 'D3') & (train_set['lodging'] == 1)].shape[0],
    train_set[(train_set['class'] == 'D3') & (train_set['stem-cankers'] == 1)].shape[0],train_set[(train_set['class'] == 'D3') & (train_set['canker-lesion'] == 1)].shape[0], train_set[(train_set['class'] == 'D3') & (train_set['fruiting-bodies'] == 1)].shape[0],train_set[(train_set['class'] == 'D3') & (train_set['external-decay'] == 1)].shape[0], train_set[(train_set['class'] == 'D3') & (train_set['mycelium'] == 1)].shape[0],
    train_set[(train_set['class'] == 'D3') & (train_set['int-discolor'] == 1)].shape[0],train_set[(train_set['class'] == 'D3') & (train_set['sclerotia'] == 1)].shape[0], train_set[(train_set['class'] == 'D3') & (train_set['fruit-pods'] == 1)].shape[0], train_set[(train_set['class'] == 'D3') & (train_set['roots'] == 1)].shape[0]],

    [train_set[(train_set['class'] == 'D3') & (train_set['date'] == 2)].shape[0],train_set[(train_set['class'] == 'D3') & (train_set['plant-stand'] == 2)].shape[0], train_set[(train_set['class'] == 'D3') & (train_set['precip'] == 2)].shape[0],train_set[(train_set['class'] == 'D3') & (train_set['temp'] == 2)].shape[0], train_set[(train_set['class'] == 'D3') & (train_set['hail'] == 2)].shape[0],
    train_set[(train_set['class'] == 'D3') & (train_set['crop-hist'] == 2)].shape[0],train_set[(train_set['class'] == 'D3') & (train_set['area-damaged'] == 2)].shape[0], train_set[(train_set['class'] == 'D3') & (train_set['severity'] == 2)].shape[0],train_set[(train_set['class'] == 'D3') & (train_set['seed-tmt'] == 2)].shape[0], train_set[(train_set['class'] == 'D3') & (train_set['germination'] == 2)].shape[0],train_set[(train_set['class'] == 'D3') & (train_set['leaves'] == 2)].shape[0], train_set[(train_set['class'] == 'D3') & (train_set['lodging'] == 2)].shape[0],
    train_set[(train_set['class'] == 'D3') & (train_set['stem-cankers'] == 2)].shape[0],train_set[(train_set['class'] == 'D3') & (train_set['canker-lesion'] == 2)].shape[0], train_set[(train_set['class'] == 'D3') & (train_set['fruiting-bodies'] == 2)].shape[0],train_set[(train_set['class'] == 'D3') & (train_set['external-decay'] == 2)].shape[0], train_set[(train_set['class'] == 'D3') & (train_set['mycelium'] == 2)].shape[0],
    train_set[(train_set['class'] == 'D3') & (train_set['int-discolor'] == 2)].shape[0],train_set[(train_set['class'] == 'D3') & (train_set['sclerotia'] == 2)].shape[0], train_set[(train_set['class'] == 'D3') & (train_set['fruit-pods'] == 2)].shape[0], train_set[(train_set['class'] == 'D3') & (train_set['roots'] == 2)].shape[0]],

    [train_set[(train_set['class'] == 'D3') & (train_set['date'] == 3)].shape[0],train_set[(train_set['class'] == 'D3') & (train_set['plant-stand'] == 3)].shape[0], train_set[(train_set['class'] == 'D3') & (train_set['precip'] == 3)].shape[0],train_set[(train_set['class'] == 'D3') & (train_set['temp'] == 3)].shape[0], train_set[(train_set['class'] == 'D3') & (train_set['hail'] == 3)].shape[0],
    train_set[(train_set['class'] == 'D3') & (train_set['crop-hist'] == 3)].shape[0],train_set[(train_set['class'] == 'D3') & (train_set['area-damaged'] == 3)].shape[0], train_set[(train_set['class'] == 'D3') & (train_set['severity'] == 3)].shape[0],train_set[(train_set['class'] == 'D3') & (train_set['seed-tmt'] == 3)].shape[0], train_set[(train_set['class'] == 'D3') & (train_set['germination'] == 3)].shape[0],train_set[(train_set['class'] == 'D3') & (train_set['leaves'] == 3)].shape[0], train_set[(train_set['class'] == 'D3') & (train_set['lodging'] == 3)].shape[0],
    train_set[(train_set['class'] == 'D3') & (train_set['stem-cankers'] == 3)].shape[0],train_set[(train_set['class'] == 'D3') & (train_set['canker-lesion'] == 3)].shape[0], train_set[(train_set['class'] == 'D3') & (train_set['fruiting-bodies'] == 3)].shape[0],train_set[(train_set['class'] == 'D3') & (train_set['external-decay'] == 3)].shape[0], train_set[(train_set['class'] == 'D3') & (train_set['mycelium'] == 3)].shape[0],
    train_set[(train_set['class'] == 'D3') & (train_set['int-discolor'] == 3)].shape[0],train_set[(train_set['class'] == 'D3') & (train_set['sclerotia'] == 3)].shape[0], train_set[(train_set['class'] == 'D3') & (train_set['fruit-pods'] == 3)].shape[0], train_set[(train_set['class'] == 'D3') & (train_set['roots'] == 3)].shape[0]],

    [train_set[(train_set['class'] == 'D3') & (train_set['date'] == 4)].shape[0],train_set[(train_set['class'] == 'D3') & (train_set['plant-stand'] == 4)].shape[0], train_set[(train_set['class'] == 'D3') & (train_set['precip'] == 4)].shape[0],train_set[(train_set['class'] == 'D3') & (train_set['temp'] == 4)].shape[0], train_set[(train_set['class'] == 'D3') & (train_set['hail'] == 4)].shape[0],
    train_set[(train_set['class'] == 'D3') & (train_set['crop-hist'] == 4)].shape[0],train_set[(train_set['class'] == 'D3') & (train_set['area-damaged'] == 4)].shape[0], train_set[(train_set['class'] == 'D3') & (train_set['severity'] == 4)].shape[0],train_set[(train_set['class'] == 'D3') & (train_set['seed-tmt'] == 4)].shape[0], train_set[(train_set['class'] == 'D3') & (train_set['germination'] == 4)].shape[0],train_set[(train_set['class'] == 'D3') & (train_set['leaves'] == 4)].shape[0], train_set[(train_set['class'] == 'D3') & (train_set['lodging'] == 4)].shape[0],
    train_set[(train_set['class'] == 'D3') & (train_set['stem-cankers'] == 4)].shape[0],train_set[(train_set['class'] == 'D3') & (train_set['canker-lesion'] == 4)].shape[0], train_set[(train_set['class'] == 'D3') & (train_set['fruiting-bodies'] == 4)].shape[0],train_set[(train_set['class'] == 'D3') & (train_set['external-decay'] == 4)].shape[0], train_set[(train_set['class'] == 'D3') & (train_set['mycelium'] == 4)].shape[0],
    train_set[(train_set['class'] == 'D3') & (train_set['int-discolor'] == 4)].shape[0],train_set[(train_set['class'] == 'D3') & (train_set['sclerotia'] == 4)].shape[0], train_set[(train_set['class'] == 'D3') & (train_set['fruit-pods'] == 4)].shape[0], train_set[(train_set['class'] == 'D3') & (train_set['roots'] == 4)].shape[0]],

    [train_set[(train_set['class'] == 'D3') & (train_set['date'] == 5)].shape[0],train_set[(train_set['class'] == 'D3') & (train_set['plant-stand'] == 5)].shape[0], train_set[(train_set['class'] == 'D3') & (train_set['precip'] == 5)].shape[0],train_set[(train_set['class'] == 'D3') & (train_set['temp'] == 5)].shape[0], train_set[(train_set['class'] == 'D3') & (train_set['hail'] == 5)].shape[0],
    train_set[(train_set['class'] == 'D3') & (train_set['crop-hist'] == 5)].shape[0],train_set[(train_set['class'] == 'D3') & (train_set['area-damaged'] == 5)].shape[0], train_set[(train_set['class'] == 'D3') & (train_set['severity'] == 5)].shape[0],train_set[(train_set['class'] == 'D3') & (train_set['seed-tmt'] == 5)].shape[0], train_set[(train_set['class'] == 'D3') & (train_set['germination'] == 5)].shape[0],train_set[(train_set['class'] == 'D3') & (train_set['leaves'] == 5)].shape[0], train_set[(train_set['class'] == 'D3') & (train_set['lodging'] == 5)].shape[0],
    train_set[(train_set['class'] == 'D3') & (train_set['stem-cankers'] == 5)].shape[0],train_set[(train_set['class'] == 'D3') & (train_set['canker-lesion'] == 5)].shape[0], train_set[(train_set['class'] == 'D3') & (train_set['fruiting-bodies'] == 5)].shape[0],train_set[(train_set['class'] == 'D3') & (train_set['external-decay'] == 5)].shape[0], train_set[(train_set['class'] == 'D3') & (train_set['mycelium'] == 5)].shape[0],
    train_set[(train_set['class'] == 'D3') & (train_set['int-discolor'] == 5)].shape[0],train_set[(train_set['class'] == 'D3') & (train_set['sclerotia'] == 5)].shape[0], train_set[(train_set['class'] == 'D3') & (train_set['fruit-pods'] == 5)].shape[0], train_set[(train_set['class'] == 'D3') & (train_set['roots'] == 5)].shape[0]],

    [train_set[(train_set['class'] == 'D3') & (train_set['date'] == 6)].shape[0],train_set[(train_set['class'] == 'D3') & (train_set['plant-stand'] == 6)].shape[0], train_set[(train_set['class'] == 'D3') & (train_set['precip'] == 6)].shape[0],train_set[(train_set['class'] == 'D3') & (train_set['temp'] == 6)].shape[0], train_set[(train_set['class'] == 'D3') & (train_set['hail'] == 6)].shape[0],
    train_set[(train_set['class'] == 'D3') & (train_set['crop-hist'] == 6)].shape[0],train_set[(train_set['class'] == 'D3') & (train_set['area-damaged'] == 6)].shape[0], train_set[(train_set['class'] == 'D3') & (train_set['severity'] == 6)].shape[0],train_set[(train_set['class'] == 'D3') & (train_set['seed-tmt'] == 6)].shape[0], train_set[(train_set['class'] == 'D3') & (train_set['germination'] == 6)].shape[0],train_set[(train_set['class'] == 'D3') & (train_set['leaves'] == 6)].shape[0], train_set[(train_set['class'] == 'D3') & (train_set['lodging'] == 6)].shape[0],
    train_set[(train_set['class'] == 'D3') & (train_set['stem-cankers'] == 6)].shape[0],train_set[(train_set['class'] == 'D3') & (train_set['canker-lesion'] == 6)].shape[0], train_set[(train_set['class'] == 'D3') & (train_set['fruiting-bodies'] == 6)].shape[0],train_set[(train_set['class'] == 'D3') & (train_set['external-decay'] == 6)].shape[0], train_set[(train_set['class'] == 'D3') & (train_set['mycelium'] == 6)].shape[0],
    train_set[(train_set['class'] == 'D3') & (train_set['int-discolor'] == 6)].shape[0],train_set[(train_set['class'] == 'D3') & (train_set['sclerotia'] == 6)].shape[0], train_set[(train_set['class'] == 'D3') & (train_set['fruit-pods'] == 6)].shape[0], train_set[(train_set['class'] == 'D3') & (train_set['roots'] == 6)].shape[0]],
    
    [train_set[(train_set['class'] == 'D4') & (train_set['date'] == 0)].shape[0],train_set[(train_set['class'] == 'D4') & (train_set['plant-stand'] == 0)].shape[0], train_set[(train_set['class'] == 'D4') & (train_set['precip'] == 0)].shape[0],train_set[(train_set['class'] == 'D4') & (train_set['temp'] == 0)].shape[0], train_set[(train_set['class'] == 'D4') & (train_set['hail'] == 0)].shape[0],
    train_set[(train_set['class'] == 'D4') & (train_set['crop-hist'] == 0)].shape[0],train_set[(train_set['class'] == 'D4') & (train_set['area-damaged'] == 0)].shape[0], train_set[(train_set['class'] == 'D4') & (train_set['severity'] == 0)].shape[0],train_set[(train_set['class'] == 'D4') & (train_set['seed-tmt'] == 0)].shape[0], train_set[(train_set['class'] == 'D4') & (train_set['germination'] == 0)].shape[0],train_set[(train_set['class'] == 'D4') & (train_set['leaves'] == 0)].shape[0], train_set[(train_set['class'] == 'D4') & (train_set['lodging'] == 0)].shape[0],
    train_set[(train_set['class'] == 'D4') & (train_set['stem-cankers'] == 0)].shape[0],train_set[(train_set['class'] == 'D4') & (train_set['canker-lesion'] == 0)].shape[0], train_set[(train_set['class'] == 'D4') & (train_set['fruiting-bodies'] == 0)].shape[0],train_set[(train_set['class'] == 'D4') & (train_set['external-decay'] == 0)].shape[0], train_set[(train_set['class'] == 'D4') & (train_set['mycelium'] == 0)].shape[0],
    train_set[(train_set['class'] == 'D4') & (train_set['int-discolor'] == 0)].shape[0],train_set[(train_set['class'] == 'D4') & (train_set['sclerotia'] == 0)].shape[0], train_set[(train_set['class'] == 'D4') & (train_set['fruit-pods'] == 0)].shape[0], train_set[(train_set['class'] == 'D4') & (train_set['roots'] == 0)].shape[0]],

    [train_set[(train_set['class'] == 'D4') & (train_set['date'] == 1)].shape[0],train_set[(train_set['class'] == 'D4') & (train_set['plant-stand'] == 1)].shape[0], train_set[(train_set['class'] == 'D4') & (train_set['precip'] == 1)].shape[0],train_set[(train_set['class'] == 'D4') & (train_set['temp'] == 1)].shape[0], train_set[(train_set['class'] == 'D4') & (train_set['hail'] == 1)].shape[0],
    train_set[(train_set['class'] == 'D4') & (train_set['crop-hist'] == 1)].shape[0],train_set[(train_set['class'] == 'D4') & (train_set['area-damaged'] == 1)].shape[0], train_set[(train_set['class'] == 'D4') & (train_set['severity'] == 1)].shape[0],train_set[(train_set['class'] == 'D4') & (train_set['seed-tmt'] == 1)].shape[0], train_set[(train_set['class'] == 'D4') & (train_set['germination'] == 1)].shape[0],train_set[(train_set['class'] == 'D4') & (train_set['leaves'] == 1)].shape[0], train_set[(train_set['class'] == 'D4') & (train_set['lodging'] == 1)].shape[0],
    train_set[(train_set['class'] == 'D4') & (train_set['stem-cankers'] == 1)].shape[0],train_set[(train_set['class'] == 'D4') & (train_set['canker-lesion'] == 1)].shape[0], train_set[(train_set['class'] == 'D4') & (train_set['fruiting-bodies'] == 1)].shape[0],train_set[(train_set['class'] == 'D4') & (train_set['external-decay'] == 1)].shape[0], train_set[(train_set['class'] == 'D4') & (train_set['mycelium'] == 1)].shape[0],
    train_set[(train_set['class'] == 'D4') & (train_set['int-discolor'] == 1)].shape[0],train_set[(train_set['class'] == 'D4') & (train_set['sclerotia'] == 1)].shape[0], train_set[(train_set['class'] == 'D4') & (train_set['fruit-pods'] == 1)].shape[0], train_set[(train_set['class'] == 'D4') & (train_set['roots'] == 1)].shape[0]],

    [train_set[(train_set['class'] == 'D4') & (train_set['date'] == 2)].shape[0],train_set[(train_set['class'] == 'D4') & (train_set['plant-stand'] == 2)].shape[0], train_set[(train_set['class'] == 'D4') & (train_set['precip'] == 2)].shape[0],train_set[(train_set['class'] == 'D4') & (train_set['temp'] == 2)].shape[0], train_set[(train_set['class'] == 'D4') & (train_set['hail'] == 2)].shape[0],
    train_set[(train_set['class'] == 'D4') & (train_set['crop-hist'] == 2)].shape[0],train_set[(train_set['class'] == 'D4') & (train_set['area-damaged'] == 2)].shape[0], train_set[(train_set['class'] == 'D4') & (train_set['severity'] == 2)].shape[0],train_set[(train_set['class'] == 'D4') & (train_set['seed-tmt'] == 2)].shape[0], train_set[(train_set['class'] == 'D4') & (train_set['germination'] == 2)].shape[0],train_set[(train_set['class'] == 'D4') & (train_set['leaves'] == 2)].shape[0], train_set[(train_set['class'] == 'D4') & (train_set['lodging'] == 2)].shape[0],
    train_set[(train_set['class'] == 'D4') & (train_set['stem-cankers'] == 2)].shape[0],train_set[(train_set['class'] == 'D4') & (train_set['canker-lesion'] == 2)].shape[0], train_set[(train_set['class'] == 'D4') & (train_set['fruiting-bodies'] == 2)].shape[0],train_set[(train_set['class'] == 'D4') & (train_set['external-decay'] == 2)].shape[0], train_set[(train_set['class'] == 'D4') & (train_set['mycelium'] == 2)].shape[0],
    train_set[(train_set['class'] == 'D4') & (train_set['int-discolor'] == 2)].shape[0],train_set[(train_set['class'] == 'D4') & (train_set['sclerotia'] == 2)].shape[0], train_set[(train_set['class'] == 'D4') & (train_set['fruit-pods'] == 2)].shape[0], train_set[(train_set['class'] == 'D4') & (train_set['roots'] == 2)].shape[0]],

    [train_set[(train_set['class'] == 'D4') & (train_set['date'] == 3)].shape[0],train_set[(train_set['class'] == 'D4') & (train_set['plant-stand'] == 3)].shape[0], train_set[(train_set['class'] == 'D4') & (train_set['precip'] == 3)].shape[0],train_set[(train_set['class'] == 'D4') & (train_set['temp'] == 3)].shape[0], train_set[(train_set['class'] == 'D4') & (train_set['hail'] == 3)].shape[0],
    train_set[(train_set['class'] == 'D4') & (train_set['crop-hist'] == 3)].shape[0],train_set[(train_set['class'] == 'D4') & (train_set['area-damaged'] == 3)].shape[0], train_set[(train_set['class'] == 'D4') & (train_set['severity'] == 3)].shape[0],train_set[(train_set['class'] == 'D4') & (train_set['seed-tmt'] == 3)].shape[0], train_set[(train_set['class'] == 'D4') & (train_set['germination'] == 3)].shape[0],train_set[(train_set['class'] == 'D4') & (train_set['leaves'] == 3)].shape[0], train_set[(train_set['class'] == 'D4') & (train_set['lodging'] == 3)].shape[0],
    train_set[(train_set['class'] == 'D4') & (train_set['stem-cankers'] == 3)].shape[0],train_set[(train_set['class'] == 'D4') & (train_set['canker-lesion'] == 3)].shape[0], train_set[(train_set['class'] == 'D4') & (train_set['fruiting-bodies'] == 3)].shape[0],train_set[(train_set['class'] == 'D4') & (train_set['external-decay'] == 3)].shape[0], train_set[(train_set['class'] == 'D4') & (train_set['mycelium'] == 3)].shape[0],
    train_set[(train_set['class'] == 'D4') & (train_set['int-discolor'] == 3)].shape[0],train_set[(train_set['class'] == 'D4') & (train_set['sclerotia'] == 3)].shape[0], train_set[(train_set['class'] == 'D4') & (train_set['fruit-pods'] == 3)].shape[0], train_set[(train_set['class'] == 'D4') & (train_set['roots'] == 3)].shape[0]],

    [train_set[(train_set['class'] == 'D4') & (train_set['date'] == 4)].shape[0],train_set[(train_set['class'] == 'D4') & (train_set['plant-stand'] == 4)].shape[0], train_set[(train_set['class'] == 'D4') & (train_set['precip'] == 4)].shape[0],train_set[(train_set['class'] == 'D4') & (train_set['temp'] == 4)].shape[0], train_set[(train_set['class'] == 'D4') & (train_set['hail'] == 4)].shape[0],
    train_set[(train_set['class'] == 'D4') & (train_set['crop-hist'] == 4)].shape[0],train_set[(train_set['class'] == 'D4') & (train_set['area-damaged'] == 4)].shape[0], train_set[(train_set['class'] == 'D4') & (train_set['severity'] == 4)].shape[0],train_set[(train_set['class'] == 'D4') & (train_set['seed-tmt'] == 4)].shape[0], train_set[(train_set['class'] == 'D4') & (train_set['germination'] == 4)].shape[0],train_set[(train_set['class'] == 'D4') & (train_set['leaves'] == 4)].shape[0], train_set[(train_set['class'] == 'D4') & (train_set['lodging'] == 4)].shape[0],
    train_set[(train_set['class'] == 'D4') & (train_set['stem-cankers'] == 4)].shape[0],train_set[(train_set['class'] == 'D4') & (train_set['canker-lesion'] == 4)].shape[0], train_set[(train_set['class'] == 'D4') & (train_set['fruiting-bodies'] == 4)].shape[0],train_set[(train_set['class'] == 'D4') & (train_set['external-decay'] == 4)].shape[0], train_set[(train_set['class'] == 'D4') & (train_set['mycelium'] == 4)].shape[0],
    train_set[(train_set['class'] == 'D4') & (train_set['int-discolor'] == 4)].shape[0],train_set[(train_set['class'] == 'D4') & (train_set['sclerotia'] == 4)].shape[0], train_set[(train_set['class'] == 'D4') & (train_set['fruit-pods'] == 4)].shape[0], train_set[(train_set['class'] == 'D4') & (train_set['roots'] == 4)].shape[0]],

    [train_set[(train_set['class'] == 'D4') & (train_set['date'] == 5)].shape[0],train_set[(train_set['class'] == 'D4') & (train_set['plant-stand'] == 5)].shape[0], train_set[(train_set['class'] == 'D4') & (train_set['precip'] == 5)].shape[0],train_set[(train_set['class'] == 'D4') & (train_set['temp'] == 5)].shape[0], train_set[(train_set['class'] == 'D4') & (train_set['hail'] == 5)].shape[0],
    train_set[(train_set['class'] == 'D4') & (train_set['crop-hist'] == 5)].shape[0],train_set[(train_set['class'] == 'D4') & (train_set['area-damaged'] == 5)].shape[0], train_set[(train_set['class'] == 'D4') & (train_set['severity'] == 5)].shape[0],train_set[(train_set['class'] == 'D4') & (train_set['seed-tmt'] == 5)].shape[0], train_set[(train_set['class'] == 'D4') & (train_set['germination'] == 5)].shape[0],train_set[(train_set['class'] == 'D4') & (train_set['leaves'] == 5)].shape[0], train_set[(train_set['class'] == 'D4') & (train_set['lodging'] == 5)].shape[0],
    train_set[(train_set['class'] == 'D4') & (train_set['stem-cankers'] == 5)].shape[0],train_set[(train_set['class'] == 'D4') & (train_set['canker-lesion'] == 5)].shape[0], train_set[(train_set['class'] == 'D4') & (train_set['fruiting-bodies'] == 5)].shape[0],train_set[(train_set['class'] == 'D4') & (train_set['external-decay'] == 5)].shape[0], train_set[(train_set['class'] == 'D4') & (train_set['mycelium'] == 5)].shape[0],
    train_set[(train_set['class'] == 'D4') & (train_set['int-discolor'] == 5)].shape[0],train_set[(train_set['class'] == 'D4') & (train_set['sclerotia'] == 5)].shape[0], train_set[(train_set['class'] == 'D4') & (train_set['fruit-pods'] == 5)].shape[0], train_set[(train_set['class'] == 'D4') & (train_set['roots'] == 5)].shape[0]],

    [train_set[(train_set['class'] == 'D4') & (train_set['date'] == 6)].shape[0],train_set[(train_set['class'] == 'D4') & (train_set['plant-stand'] == 6)].shape[0], train_set[(train_set['class'] == 'D4') & (train_set['precip'] == 6)].shape[0],train_set[(train_set['class'] == 'D4') & (train_set['temp'] == 6)].shape[0], train_set[(train_set['class'] == 'D4') & (train_set['hail'] == 6)].shape[0],
    train_set[(train_set['class'] == 'D4') & (train_set['crop-hist'] == 6)].shape[0],train_set[(train_set['class'] == 'D4') & (train_set['area-damaged'] == 6)].shape[0], train_set[(train_set['class'] == 'D4') & (train_set['severity'] == 6)].shape[0],train_set[(train_set['class'] == 'D4') & (train_set['seed-tmt'] == 6)].shape[0], train_set[(train_set['class'] == 'D4') & (train_set['germination'] == 6)].shape[0],train_set[(train_set['class'] == 'D4') & (train_set['leaves'] == 6)].shape[0], train_set[(train_set['class'] == 'D4') & (train_set['lodging'] == 6)].shape[0],
    train_set[(train_set['class'] == 'D4') & (train_set['stem-cankers'] == 6)].shape[0],train_set[(train_set['class'] == 'D4') & (train_set['canker-lesion'] == 6)].shape[0], train_set[(train_set['class'] == 'D4') & (train_set['fruiting-bodies'] == 6)].shape[0],train_set[(train_set['class'] == 'D4') & (train_set['external-decay'] == 6)].shape[0], train_set[(train_set['class'] == 'D4') & (train_set['mycelium'] == 6)].shape[0],
    train_set[(train_set['class'] == 'D4') & (train_set['int-discolor'] == 6)].shape[0],train_set[(train_set['class'] == 'D4') & (train_set['sclerotia'] == 6)].shape[0], train_set[(train_set['class'] == 'D4') & (train_set['fruit-pods'] == 6)].shape[0], train_set[(train_set['class'] == 'D4') & (train_set['roots'] == 6)].shape[0]]
]
    soy_set_frequency_table = pd.DataFrame(data_struct, columns=["date","plant-stand","precip","temp","hail","crop-hist","area-damaged","severity","seed-tmt","germination","leaves","lodging","stem-cankers","canker-lesion","fruiting-bodies","external-decay","mycelium","int-discolor","sclerotia","fruit-pods","roots",],index=['0-D1','1-D1','2-D1','3-D1','4-D1','5-D1','6-D1','0-D2','1-D2','2-D2','3-D2','4-D2','5-D2','6-D2','0-D3','1-D3','2-D3','3-D3','4-D3','5-D3','6-D3','0-D4','1-D4','2-D4','3-D4','4-D4','5-D4','6-D4'])
    
    return soy_set_frequency_table

def binning(dataset, bins, labels):
    print("glass---------------")	
    for col_name, col_data in dataset.iteritems():
        labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        dataset[col_name] = pd.cut(dataset[col_name], 10, labels=labels)
        print(dataset)              											
    print(dataset)


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

def calculate_feature_product(test_set, train_set_likelihood_table, train_type_dict):
    i = 1
    for row_index, row in test_set.iterrows():
        f_product_2 = 1;
        f_product_4 = 1;
        for col_idx, value in row.items():
            if col_idx != "Sample code number" and col_idx != "class":
                for t_key in train_type_dict:
                    if t_key == 2:
                        label = str(value) + "-" + str(t_key)
                        if i == 1:
                            print('Value:',train_set_likelihood_table.loc[label, col_idx])
                        f_product_2 *= train_set_likelihood_table.loc[label, col_idx]
                        if i == 1:
                            print('F prd 2:', f_product_2)
                            print()
                    if t_key == 4:
                        label = str(value) + "-" + str(t_key)
                        if i == 1:
                            print('Value:',train_set_likelihood_table.loc[label, col_idx])
                        f_product_4 *= train_set_likelihood_table.loc[label, col_idx]
                        if i == 1:
                            print('F prd 4:', f_product_4)
                            print()
        c_2 = f_product_2 * train_type_dict[2]
        c_4 = f_product_4 * train_type_dict[4]
        if i == 1:
            print("FInal f2:",c_2)
            print("FInal f4:",c_4)
            i = 2

        if c_2 > c_4:
            test_set["Classifier"] = 2
        else:
            test_set["Classifier"] = 4
        test_set.to_csv('classes.csv')
        print(c_2,",",c_4)
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
    # print(class_fold_set)
        flag = False
    for key,value in class_fold_set.items():
        if flag == False:
            fold1,fold2,fold3,fold4,fold5,fold6,fold7,fold8,fold9,fold10 = value[0],value[1],value[2],value[3],value[4],value[5],value[6],value[7],value[8],value[9]
            flag = True
        else:
            fold1 = pd.concat([fold1,value[0]])
            print("-------------fold1----------")
            print(fold1)
            fold2 = pd.concat([fold1,value[1]])
            fold3 = pd.concat([fold1,value[2]])
            fold4 = pd.concat([fold1,value[3]])
            fold5 = pd.concat([fold1,value[4]])
            fold6 = pd.concat([fold1,value[5]])
            fold7 = pd.concat([fold1,value[6]])
            fold8 = pd.concat([fold1,value[7]])
            fold9 = pd.concat([fold1,value[8]])
            fold10 = pd.concat([fold1,value[9]])
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
def create_noise_set(dataframe):
    # select 10% of features at random
    # shuffle their values
    # return the resulting dataframe
    return None

#function to create confusion matrix 
def calculate_loss_function(fold, class_names):
    confusion_matrix = {}
    loss = {}
    actual_class = fold["class"].tolist()
    guessed_class = fold["guess"].tolist()
    for name in class_names:
        confusion_matrix[name] = {"TP":0, "FP":0, "FN": 0, "TN":0}
        index = 0
        for act in actual_class:
            if name == guessed_class[index] and name == act:
                predicated = "TP"
            if name == guessed_class[index] and name != act:
                predicated = "FP"
            if name != guessed_class[index] and name == act:
                predicated = "FN"
            if name != guessed_class[index] and name != act:
                predicated = "TN"
            confusion_matrix[name][predicated] += 1
            index += 1

    for name in class_names:
        loss[name] = {"Accuracy": 0, "F1":0}
        total = confusion_matrix[name]["TP"] + confusion_matrix[name]["FP"] + confusion_matrix[name]["TP"] + confusion_matrix[name]["TN"]
        percision = confusion_matrix[name]["TP"] / (confusion_matrix[name]["TP"] + confusion_matrix[name]["FP"])
        recall = confusion_matrix[name]["TP"] / (confusion_matrix[name]["TP"] + confusion_matrix[name]["TN"])
        F1 = 2 * ((percision * recall) / (percision + recall))
        accuracy =   (confusion_matrix[name]["TP"] + confusion_matrix[name]["TN"]) / total
        loss[name]["Accuracy"] = accuracy
        loss[name]["F1"] = F1
    return loss
    
def getClassTypes(dataset):
    class_names = dataset["Class"]
    class_names = class_names.unique()
    return class_names

if __name__ == '__main__':
    # import data into dataframes
    cancer_df = pd.read_csv("breast-cancer-wisconsin-cleaned.txt", sep=",", header=None)
    # glass_df = pd.read_csv("glass.data", sep=",", header=None)
    votes_df = pd.read_csv("house-votes-84.data", sep=",", header=None)
    # iris_df = pd.read_csv("iris.data", sep=",", header=None)
    soy_df = pd.read_csv("soybean-small-cleaned.csv", sep=",", header=None)
    glass_df = pd.read_csv("glass.data", sep=",", header=None)
    #bin_glass_set(glass_df)

    # Binning --> Change from continuous to discrete values
    ########################
    # FILL ME OUT
    #######################

    # Create Noise Data Sets
    cancer_noise_df=create_noise_set(cancer_df)
    # glass_noise_df=create_noise_set(glass_df)
    votes_noise_df=create_noise_set(votes_df)
    # iris_noise_df=create_noise_set(iris_df)
    soy_noise_df=create_noise_set(soy_df)

    # label original data frames
    cancer_df.columns = ["Sample code number", "Uniformity of Clump Thickness", "Uniformity of Cell Size", "Cell Shape", "Marginal Adhesion", "Single Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses", "class"]
    #glass_df.columns = ['id-num','retractive-index','sodium','magnesium','aluminum','silicon','potasium','calcium','barium','iron','class']
    votes_df.columns = ["class","infants","water","adoption","physician","salvador","religious","satellite","nicaragua","missile","immigration","synfuels","education","superfund","crime","exports","south-africa"]
    # iris_df.columns = ['sepal-length','sepal-width','petal-length','petal-width','class']
    soy_df.columns = ["date","plant-stand","precip","temp","hail","crop-hist","area-damaged","severity","seed-tmt","germination","leaves","lodging","stem-cankers","canker-lesion","fruiting-bodies","external-decay","mycelium","int-discolor","sclerotia","fruit-pods","roots","class"]

    #label noise data frames
    # cancer_noise_df.columns = ["Sample code number", "Uniformity of Clump Thickness", "Uniformity of Cell Size", "Cell Shape", "Marginal Adhesion", "Single Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses", "class"]
    # # glass_noise_df.columns = ['id-num','retractive-index','sodium','magnesium','aluminum','silicon','potasium','calcium','barium','iron','class']
    # votes_noise_df.columns = ["class","infants","water","adoption","physician","salvador","religious","satellite","nicaragua","missile","immigration","synfuels","education","superfund","crime","exports","south-africa"]
    # # iris_noise_df.columns = ['sepal-length','sepal-width','petal-length','petal-width','class']
    # soy_noise_df.columns = ["date","plant-stand","precip","temp","hail","crop-hist","area-damaged","severity","seed-tmt","germination","leaves","lodging","stem-cankers","canker-lesion","fruiting-bodies","external-decay","mycelium","int-discolor","sclerotia","fruit-pods","roots","class"]

    print("IMPORTED TABLES")
    print(cancer_df)
    # print(glass_df)
    print(votes_df)
    # print(iris_df)
    print(soy_df)
    print("--------------------------------")

    # Calculate variables for later reference
    cancer_classes = cancer_df['class'].unique()
    # glass_classes = cancer_df['class'].unique()
    votes_classes = cancer_df['class'].unique()
    # iris_classes = cancer_df['class'].unique()
    soy_classes = cancer_df['class'].unique()

    # Create Original training and testing dataframes
    cancer_training1,cancer_testing1,cancer_training2,cancer_testing2,cancer_training3,cancer_testing3,cancer_training4,cancer_testing4,cancer_training5,cancer_testing5,cancer_training6,cancer_testing6,cancer_training7,cancer_testing7,cancer_training8,cancer_testing8,cancer_training9,cancer_testing9,cancer_training10,cancer_testing10 = stratify_and_fold(cancer_df)
    # glass_training1,glass_testing1,glass_training2,glass_testing2,glass_training3,glass_testing3,glass_training4,glass_testing4,glass_training5,glass_testing5,glass_training6,glass_testing6,glass_training7,glass_testing7,glass_training8,glass_testing8,glass_training9,glass_testing9,glass_training10,glass_testing10 = stratify_and_fold(glass_df)
    votes_training1,votes_testing1,votes_training2,votes_testing2,votes_training3,votes_testing3,votes_training4,votes_testing4,votes_training5,votes_testing5,votes_training6,votes_testing6,votes_training7,votes_testing7,votes_training8,votes_testing8,votes_training9,votes_testing9,votes_training10,votes_testing10 = stratify_and_fold(votes_df)
    # iris_training1,iris_testing1,iris_training2,iris_testing2,iris_training3,iris_testing3,iris_training4,iris_testing4,iris_training5,iris_testing5,iris_training6,iris_testing6,iris_training7,iris_testing7,iris_training8,iris_testing8,iris_training9,iris_testing9,iris_training10,iris_testing10 = stratify_and_fold(iris_df)
    soy_training1,soy_testing1,soy_training2,soy_testing2,soy_training3,soy_testing3,soy_training4,soy_testing4,soy_training5,soy_testing5,soy_training6,soy_testing6,soy_training7,soy_testing7,soy_training8,soy_testing8,soy_training9,soy_testing9,soy_training10,soy_testing10 = stratify_and_fold(soy_df)

    # Create Noise training and testing dataframes
    # cancer_noise_training1,cancer_noise_testing1,cancer_noise_training2,cancer_noise_testing2,cancer_noise_training3,cancer_noise_testing3,cancer_noise_training4,cancer_noise_testing4,cancer_noise_training5,cancer_noise_testing5,cancer_noise_training6,cancer_noise_testing6,cancer_noise_training7,cancer_noise_testing7,cancer_noise_training8,cancer_noise_testing8,cancer_noise_training9,cancer_noise_testing9,cancer_noise_training10,cancer_noise_testing10 = stratify_and_fold(cancer_noise_df)
    # # glass_noise_training1,glass_noise_testing1,glass_noise_training2,glass_noise_testing2,glass_noise_training3,glass_noise_testing3,glass_noise_training4,glass_noise_testing4,glass_noise_training5,glass_noise_testing5,glass_noise_training6,glass_noise_testing6,glass_noise_training7,glass_noise_testing7,glass_noise_training8,glass_noise_testing8,glass_noise_training9,glass_noise_testing9,glass_noise_training10,glass_noise_testing10 = stratify_and_fold(glass_noise_df)
    # votes_noise_training1,votes_noise_testing1,votes_noise_training2,votes_noise_testing2,votes_noise_training3,votes_noise_testing3,votes_noise_training4,votes_noise_testing4,votes_noise_training5,votes_noise_testing5,votes_noise_training6,votes_noise_testing6,votes_noise_training7,votes_noise_testing7,votes_noise_training8,votes_noise_testing8,votes_noise_training9,votes_noise_testing9,votes_noise_training10,votes_noise_testing10 = stratify_and_fold(votes_noise_df)
    # iris_noise_training1,iris_noise_testing1,iris_noise_training2,iris_noise_testing2,iris_noise_training3,iris_noise_testing3,iris_noise_training4,iris_noise_testing4,iris_noise_training5,iris_noise_testing5,iris_noise_training6,iris_noise_testing6,iris_noise_training7,iris_noise_testing7,iris_noise_training8,iris_noise_testing8,iris_noise_training9,iris_noise_testing9,iris_noise_training10,iris_noise_testing10 = stratify_and_fold(iris_noise_df)
    # soy_noise_training1,soy_noise_testing1,soy_noise_training2,soy_noise_testing2,soy_noise_training3,soy_noise_testing3,soy_noise_training4,soy_noise_testing4,soy_noise_training5,soy_noise_testing5,soy_noise_training6,soy_noise_testing6,soy_noise_training7,soy_noise_testing7,soy_noise_training8,soy_noise_testing8,soy_noise_training9,soy_noise_testing9,soy_noise_training10,soy_noise_testing10 = stratify_and_fold(soy_noise_df)

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

    print("LIKELIHOOD TABLES")
    # Calculate Original Likelihood Tables
    cancer_train1_likehood_table = calculate_likelihood_table(cancer_training1, cancer_train1_frequency_table,cancer_classes)
    print("RETURNED")
    print(cancer_train1_likehood_table)
    cancer_train2_likehood_table = calculate_likelihood_table(cancer_training2, cancer_train2_frequency_table,cancer_classes)
    cancer_train3_likehood_table = calculate_likelihood_table(cancer_training3, cancer_train3_frequency_table,cancer_classes)
    cancer_train4_likehood_table = calculate_likelihood_table(cancer_training4, cancer_train4_frequency_table,cancer_classes)
    cancer_train5_likehood_table = calculate_likelihood_table(cancer_training5, cancer_train5_frequency_table,cancer_classes)
    cancer_train6_likehood_table = calculate_likelihood_table(cancer_training6, cancer_train6_frequency_table,cancer_classes)
    cancer_train7_likehood_table = calculate_likelihood_table(cancer_training7, cancer_train7_frequency_table,cancer_classes)
    cancer_train8_likehood_table = calculate_likelihood_table(cancer_training8, cancer_train8_frequency_table,cancer_classes)
    cancer_train9_likehood_table = calculate_likelihood_table(cancer_training9, cancer_train9_frequency_table,cancer_classes)
    cancer_train10_likehood_table = calculate_likelihood_table(cancer_training10, cancer_train10_frequency_table,cancer_classes)
    
    votes_train1_likehood_table = calculate_likelihood_table(votes_training1, votes_train1_frequency_table,votes_classes)
    print("RETURNED")
    print(votes_train1_likehood_table)
    votes_train2_likehood_table = calculate_likelihood_table(votes_training2, votes_train2_frequency_table,votes_classes)
    votes_train3_likehood_table = calculate_likelihood_table(votes_training3, votes_train3_frequency_table,votes_classes)
    votes_train4_likehood_table = calculate_likelihood_table(votes_training4, votes_train4_frequency_table,votes_classes)
    votes_train5_likehood_table = calculate_likelihood_table(votes_training5, votes_train5_frequency_table,votes_classes)
    votes_train6_likehood_table = calculate_likelihood_table(votes_training6, votes_train6_frequency_table,votes_classes)
    votes_train7_likehood_table = calculate_likelihood_table(votes_training7, votes_train7_frequency_table,votes_classes)
    votes_train8_likehood_table = calculate_likelihood_table(votes_training8, votes_train8_frequency_table,votes_classes)
    votes_train9_likehood_table = calculate_likelihood_table(votes_training9, votes_train9_frequency_table,votes_classes)
    votes_train10_likehood_table = calculate_likelihood_table(votes_training10, votes_train10_frequency_table,votes_classes)
    
    soy_train1_likehood_table = calculate_likelihood_table(soy_training1, soy_train1_frequency_table,soy_classes)
    print("RETURNED")
    print(soy_train1_likehood_table)
    soy_train2_likehood_table = calculate_likelihood_table(soy_training2, soy_train2_frequency_table,soy_classes)
    soy_train3_likehood_table = calculate_likelihood_table(soy_training3, soy_train3_frequency_table,soy_classes)
    soy_train4_likehood_table = calculate_likelihood_table(soy_training4, soy_train4_frequency_table,soy_classes)
    soy_train5_likehood_table = calculate_likelihood_table(soy_training5, soy_train5_frequency_table,soy_classes)
    soy_train6_likehood_table = calculate_likelihood_table(soy_training6, soy_train6_frequency_table,soy_classes)
    soy_train7_likehood_table = calculate_likelihood_table(soy_training7, soy_train7_frequency_table,soy_classes)
    soy_train8_likehood_table = calculate_likelihood_table(soy_training8, soy_train8_frequency_table,soy_classes)
    soy_train9_likehood_table = calculate_likelihood_table(soy_training9, soy_train9_frequency_table,soy_classes)
    soy_train10_likehood_table = calculate_likelihood_table(soy_training10, soy_train10_frequency_table,soy_classes)
    print("------------------------------------------------")
