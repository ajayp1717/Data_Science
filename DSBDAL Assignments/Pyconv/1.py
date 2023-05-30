# %%
import pandas as pd
import numpy as np


# %%
df=pd.read_csv("AQI.csv")

# %%
df.head()

# %%
df.info()

# %%
df.describe()

# %%
df.isnull()

# %%
df.isnull().sum()

# %%
df.tail()

# %%
df.dtypes

# %%
df['AQI Value']=df['AQI Value'].astype('float64')

# %%
df.dtypes


# %%


# %%
df.shape

# %%
