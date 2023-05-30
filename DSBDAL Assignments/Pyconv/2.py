# %%
import pandas as pd
import numpy as np

# %%
df=pd.read_csv('AQI.csv')

# %%
df.head()

# %%
df.tail()

# %%
df.describe()

# %%
df.dtypes

# %%
df.shape

# %%
df.info()

# %%
df.isnull().sum()

# %%
df

# %%
from sklearn.preprocessing import LabelEncoder

# %%
categorical_column=['Country','City']
label_encoder=LabelEncoder()

# %%
for column in categorical_column:
    df[column + '_encoded'] = label_encoder.fit_transform(df[column])

# %%
df

# %%
