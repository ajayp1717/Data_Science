# %%
"""
### Import the required Libraries
"""

# %%
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd  
import seaborn as sns 

%matplotlib inline

# %%
pip install scikit-learn==1.1.3

# %%
"""
### Load the Boston Housing DataSet from scikit-learn
"""

# %%
from sklearn.datasets import load_boston
boston_dataset = load_boston()

# %%
print(boston_dataset.keys())

# %%
"""
### Load the data into pandas dataframe
"""

# %%
boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
boston.head()

# %%
boston['MEDV'] = boston_dataset.target

# %%
"""
### Data preprocessing
"""

# %%
boston.isnull().sum()

# %%
"""
### Data Visualization
"""

# %%
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.distplot(boston['MEDV'], bins=30)
plt.show()

# %%
"""
### Correlation matrix
"""

# %%
correlation_matrix = boston.corr().round(2)
# annot = True to print the values inside the square
sns.heatmap(data=correlation_matrix, annot=True)

# %%
"""
#### Observation: MEDV is strongly correlated to LSTAT, RM
"""

# %%
plt.scatter(x=boston['RM'],y=boston['MEDV'])
plt.xlabel('RM')
plt.ylabel('MEDV')
#prices increase with increase in RM

# %%
plt.scatter(x=boston['LSTAT'],y=boston['MEDV'])
plt.xlabel('LSTAT')
plt.ylabel('MEDV')
#prices decrease with increase in LSTAT

# %%
"""
### Prepare the data for training
"""

# %%
X = pd.DataFrame(np.c_[boston['LSTAT'], boston['RM']], columns = ['LSTAT','RM'])
Y = boston['MEDV']

# %%
#normalizing the data
normalized_X=(X-X.min())/(X.max()-X.min())
normalized_Y=(Y-Y.min())/(Y.max()-Y.min())

# %%
"""
### Spliting the data into training and testing sets
"""

# %%
#80% for training and 20% for testing
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=10)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

# %%
#80% for training and 20% for testing after normalization
from sklearn.model_selection import train_test_split

Xn_train, Xn_test, Yn_train, Yn_test = train_test_split(normalized_X, normalized_Y, test_size = 0.2, random_state=10)
print(Xn_train.shape)
print(Xn_test.shape)
print(Yn_train.shape)
print(Yn_test.shape)

# %%
"""
### Training the model using sklearn LinearRegression
"""

# %%
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

lin_model = LinearRegression()
lin_model.fit(X_train, Y_train)

# %%
#after normalization
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

n_lin_model = LinearRegression()
n_lin_model.fit(Xn_train, Yn_train)

# %%
#To retrieve the intercept:
print(lin_model.intercept_)

#For retrieving the slope:
print(lin_model.coef_)

# %%
#after normalization

#To retrieve the intercept:
print(n_lin_model.intercept_)

#For retrieving the slope:
print(n_lin_model.coef_)

# %%
"""
### Model Evaluation
"""

# %%
# model evaluation for training set
y_train_predict = lin_model.predict(X_train)
rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))
mse= mean_squared_error(Y_train, y_train_predict)
mae= mean_absolute_error(Y_train, y_train_predict)
print("The model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('MSE is {}'.format(mse))
print('MAE is {}'.format(mae))
print("\n")

# model evaluation for testing set
y_test_predict = lin_model.predict(X_test)
rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
mse =(mean_squared_error(Y_test, y_test_predict))
mae= mean_absolute_error(Y_test, y_test_predict)
print("The model performance for testing set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('MSE is {}'.format(mse))
print('MAE is {}'.format(mae))

# %%
#after normalization

# model evaluation for training set
n_y_train_predict = n_lin_model.predict(Xn_train)
n_rmse = (np.sqrt(mean_squared_error(Yn_train, n_y_train_predict)))
n_mse= (mean_squared_error(Yn_train, n_y_train_predict))
n_mae= mean_absolute_error(Yn_train, n_y_train_predict)
print("The model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(n_rmse))
print('MSE is {}'.format(n_mse))
print('MAE is {}'.format(n_mae))
print("\n")

# model evaluation for testing set
n_y_test_predict = n_lin_model.predict(Xn_test)
n_rmse = (np.sqrt(mean_squared_error(Yn_test, n_y_test_predict)))
n_mse =(mean_squared_error(Yn_test, n_y_test_predict))
n_mae= mean_absolute_error(Yn_test, n_y_test_predict)
print("The model performance for testing set")
print("--------------------------------------")
print('RMSE is {}'.format(n_rmse))
print('MSE is {}'.format(n_mse))
print('MSE is {}'.format(n_mae))


# %%
# plotting the y_test vs y_pred
plt.scatter(Y_test, y_test_predict, color='gray')
plt.show()
# ideally should have been a straight line

# %%
#after normalization

# plotting the y_test vs y_pred
plt.scatter(Yn_test, n_y_test_predict, color='gray')
plt.show()
# ideally should have been a straight line

# %%
"""
### Gradient Descent
"""

# %%
# Building the model for LSTAT vs MEDV
m = 0
c = 0

L = 0.0001  # The learning Rate
epochs = 1000  # The number of iterations to perform gradient descent
X=boston['LSTAT']

n = float(len(X)) # Number of elements in X

# Performing Gradient Descent 
for i in range(epochs): 
    Y_pred = m*X + c  # The current predicted value of Y
    D_m = (-2/n) * sum(X * (Y - Y_pred))  # Derivative wrt m
    D_c = (-2/n) * sum(Y - Y_pred)  # Derivative wrt c
    m = m - L * D_m  # Update m
    c = c - L * D_c  # Update c
    
print (m, c)

# %%
# Making predictions
Y_pred = m*X + c

plt.scatter(X, Y) 
plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='red')  # regression line
plt.show()

# %%
