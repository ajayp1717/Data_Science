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
# Get the feature names and their types
feature_types = df.dtypes
print("Features and their types:")
print(feature_types)

# %%


# Plot histograms for each feature
df.hist(figsize=(10, 8))
plt.suptitle('Histograms of Features in Titanic Dataset')
plt.tight_layout()
plt.show()

# %%
# Plot box plots for each feature
plt.figure(figsize=(10, 8))
sns.boxplot(data=df, orient='h')
plt.title('Box Plots of Features in Titanic Dataset')
plt.xlabel('Values')
plt.show()

# %%
# Identify and highlight outliers
outliers = {}
for column in df.columns:
    if df[column].dtype != 'object':
        # Calculate the interquartile range (IQR)
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        # Determine the outlier thresholds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Find the outliers for the current feature
        feature_outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        outliers[column] = feature_outliers

        # Highlight the outliers in the plot
        outlier_indexes = feature_outliers.index.tolist()
        plt.plot(feature_outliers[column], outlier_indexes, 'ro', markersize=5)
plt.show()


# %%
# Display the outliers
print("Outliers:")
for column, outlier_data in outliers.items():
    if not outlier_data.empty:
        print(f"\nFeature: {column}")
        print(outlier_data)


# %%
