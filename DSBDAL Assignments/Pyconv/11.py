# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# %%
df=sns.load_dataset('titanic')

# %%
df.dtypes

# %%
df.head()

# %%
# Filter the dataset to remove missing values in the 'age' column
filtered_df = df.dropna(subset=['age'])

# %%
# Plot the box plot
plt.figure(figsize=(8, 6))
sns.boxplot(x='sex', y='age', hue='survived', data=filtered_df)
plt.title('Distribution of Age with respect to Gender and Survival Status')
plt.xlabel('Gender')
plt.ylabel('Age')
plt.show()

# %%
# Summary statistics
summary_stats = filtered_df.groupby(['sex', 'survived'])['age'].agg(['count', 'mean','min','max', 'std'])

print("Summary Statistics:\n")
print(summary_stats)

# %%
