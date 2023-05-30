# %%
"""
# Importing required libraries
"""

# %%
#importing required libraries
import pandas as pd
import numpy as np
import statistics as st 

# %%
#reading the dataset
df = pd.read_csv("Iris.csv")

# %%
df

# %%
df.info()

# %%
df.dtypes

# %%
"""
# Creating a list of numeric values for each categorical variable
"""

# %%
df["Species"].unique()

# %%
#grouping the data based on the species
group=df.groupby(df["Species"])

# %%
setosa=group.get_group("Iris-setosa")
versicolor=group.get_group("Iris-versicolor")
virginica=group.get_group("Iris-virginica")

# %%
setosa

# %%
versicolor

# %%
virginica

# %%
setosa.describe()

# %%
versicolor.describe()

# %%
virginica.describe()

# %%
"""
# Calculating statistical details 
"""

# %%
#function to calculate mean
def cal_mean(data):
    return sum(data)/len(data)

# %%
cal_mean(setosa["SepalLengthCm"])

# %%
cal_mean(versicolor["SepalWidthCm"])

# %%
cal_mean(virginica["PetalLengthCm"])

# %%
virginica["PetalLengthCm"].mean()

# %%
#function to calculate median
def cal_median(data):
    n=len(data)
    index = n // 2
    if n % 2:
         return sorted(data)[index]
    return sum(sorted(data)[index - 1:index + 1]) / 2

# %%
cal_median(setosa["SepalLengthCm"])

# %%
cal_median(versicolor["SepalWidthCm"])

# %%
cal_median(virginica["PetalLengthCm"])

# %%
virginica["PetalLengthCm"].median()

# %%
#function to calculate mode
from collections import Counter
def cal_mode(data):
     c = Counter(data)
     return [k for k, v in c.items() if v == c.most_common(1)[0][1]]

# %%
cal_mode(setosa["SepalLengthCm"])

# %%
cal_mode(versicolor["SepalWidthCm"])

# %%
cal_mode(virginica["PetalLengthCm"])

# %%
virginica["PetalLengthCm"].mode()

# %%
#function to calculate standard deviation
import math
def cal_std(data):
    sm=0
    for i in range(len(data)):
       sm+=data[i]
       mean = sm/len(data)

    deviation_sum = 0
    for i in range(len(data)):
       deviation_sum+=(data[i]- mean)**2
    
    #calculating population standard deviation of the dataset
    psd = math.sqrt((deviation_sum)/len(data))

    #calculating sample standard deviation of the dataset
    #ssd = math.sqrt((deviation_sum)/len(data) - 1)
    return psd

# %%
cal_std(setosa["SepalLengthCm"])

# %%
cal_std(setosa["SepalWidthCm"])

# %%
cal_std(setosa["PetalWidthCm"])

# %%
setosa["PetalWidthCm"].std()

# %%
#funtion to calculate percentile
import math

def cal_percentile(input, q):
    data_sorted = sorted(input) # Sort in ascending order
    
    index = math.ceil(q / 100 * len(data_sorted))

    return data_sorted[index]

# %%
cal_percentile(setosa["SepalWidthCm"],25)

# %%
cal_percentile(setosa["PetalWidthCm"],50)

# %%
cal_percentile(versicolor["SepalWidthCm"],75)

# %%
np.percentile(versicolor["SepalWidthCm"],75)

# %%
#function to calculate minimum
def cal_min(df):
    l=list(map(float,df))
    min1=l[0]
    for i in range(1,len(l)):
        if(l[i]<min1):
            min1=l[i]
    return min1

# %%
cal_min(setosa["SepalWidthCm"])

# %%
cal_min(setosa["PetalWidthCm"])

# %%
cal_min(versicolor["SepalWidthCm"])

# %%
versicolor["SepalWidthCm"].min()

# %%
#function to calculate maximum
def cal_max(df):
    l=list(map(float,df))
    max1=l[0]
    for i in range(1,len(l)):
        if(l[i]>max1):
            max1=l[i]
    return max1

# %%
cal_max(setosa["PetalWidthCm"])

# %%
cal_max(versicolor["SepalWidthCm"])

# %%
cal_max(setosa["SepalWidthCm"])

# %%
setosa["SepalWidthCm"].max()

# %%
