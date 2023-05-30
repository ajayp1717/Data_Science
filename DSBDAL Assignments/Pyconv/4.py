# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
df=pd.read_csv('StudentsPerformance_modified.csv')

# %%
df.info()

# %%
df.describe()

# %%
df.drop('grade',axis='columns',inplace=True)

# %%
numeric_vars = df.select_dtypes(include=np.number).columns


# %%
# # Create a box plot for each numeric variable
# for var in numeric_vars:
    
plt.figure()
sns.boxplot(df['math score'])
plt.title("Box plot for math score")
plt.show()




# %%
q1=np.percentile(df['math score'],25,interpolation="midpoint")


    

# %%
q3=np.percentile(df['math score'],75,interpolation="midpoint")

# %%
iqr=q3-q1
up=q3+1.5*iqr
low=q1-1.5*iqr

# %%
uplier=np.where(df['math score']>=up)
df.drop(uplier[0],inplace=True)

# %%
print(np.where(df["math score"]>=up))

# %%
lowlier=np.where(df['math score']<=low)
df.drop(lowlier[0],inplace=True)

# %%
print(np.where(df['math score']<=low))

# %%

sns.boxplot(df['math score'])


# %%
# Apply min-max normalization to each numeric variable

min_val = df['math score'].min()
max_val = df['math score'].max()
df['math score'] = (df['math score'] - min_val) / (max_val - min_val)


# %%
# Apply z-score normalization to each numeric variable

mean_val = df['math score'].mean()
std_val = df['math score'].std()
df['math score'] = (df['math score'] - mean_val) / std_val  

# %%
sns.boxplot(df['math score'])

# %%
