# %%
import pandas as pd
import numpy as np

# %%
df=pd.read_csv('StudentsPerformance.csv')

# %%
df.info()

# %%
df.isnull().sum()

# %%
df.describe()

# %%
"""
Handling Inconsistencies and missing values

"""

# %%
newdf=df.dropna(axis='rows')    #drop rows with missing values
# 

# %%
newdf.dropna(axis='columns')    #drop columns with missing values

# %%
newdf.shape

# %%
df['math score'].fillna(df['math score'].mode(), inplace=True)          #filling na with mode

# %%
df

# %%
df.isnull().sum()

# %%
"""
Data  Transformations
"""

# %%
#Scaling a variable:
from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler(feature_range=(0,10))

# %%
df['scaled_reading score'] = scaler.fit_transform(df[['reading score']])

# %%
df

# %%
#Non-Linear to Linear

import numpy as np

# Assuming 'variable' is the column you want to transform
df['transformed_variable'] = np.log(df['variable'])

# %%
