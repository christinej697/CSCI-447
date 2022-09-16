import pandas as pd

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