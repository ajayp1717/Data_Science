# %%
import pandas as pd
import numpy as np

# Read the dataset from a CSV file
df = pd.read_csv('AQI.csv')

# Calculate summary statistics grouped by a categorical variable


# %%
df.info()

# %%
df.describe()

# %%
df.dtypes

# %%
# User-defined function for mean
def cal_mean(data):
    return sum(data) / len(data)

# %%
def cal_median(data):
    n=len(data)
    index = n // 2
    if n % 2:
        return sum(sorted(data)[index - 1:index + 1]) / 2
         
    return sorted(data)[index]
    

# %%
def cal_mode(x):
    from collections import Counter
    y=Counter(x).most_common(1)[0][0]
    return y

# %%
def cal_std(data):
    mean = cal_mean(data)
    squared_diffs = [(x - mean) ** 2 for x in data]
    variance = sum(squared_diffs) / len(data)
    return variance ** 0.5

# %%
def cal_percentile(i, q):
    data_sorted = sorted(i) # Sort in ascending order
    
    index = math.ceil(q / 100 * len(data_sorted))

    return data_sorted[index]

# %%
def cal_min(data):
    l=list(data)
    l.sort()
    return l[0]

# %%
def cal_max(data):
    l=list(data)
    l.sort(reverse=True)
    return l[0]

# %%
grouped_stats = df.groupby('Country')['AQI Value'].agg(['mean','median', 'min', 'max', 'std'])

# Create a list of summary statistics for each category
summary_list = grouped_stats.values.tolist()

# Print the summary statistics


# %%
print(grouped_stats)
print(summary_list)

# %%
