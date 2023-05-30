# %%
"""
### Import Libraries
"""

# %%
#importing the required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %%
"""
### Load Dataset
"""

# %%
#loading the dataset
df = pd.read_csv('Social_Network_Ads.csv')
df.head()

# %%
"""
### Select x and y
"""

# %%
#selecting X and Y
X = df.iloc[:, [2, 3]].values
Y = df.iloc[:, 4].values

print(X[:3, :])
print('-'*15)
print(Y[:3])

# %%
"""
### Split the data
"""

# %%
#splitting the data as 75% for training and 25% for testing
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

print(X_train[:3])
print('-'*15)
print(Y_train[:3])
print('-'*15)
print(X_test[:3])
print('-'*15)
print(Y_test[:3])

# %%
"""
### data preprocessing
"""

# %%
#data preprocessing
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

print(X_train[:3])
print('-'*15)
print(X_test[:3])

# %%
"""
### Logistic Regression
"""

# %%
#build logistic regression model
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0, solver='lbfgs' )
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)

print(X_test[:10])
print('-'*15)
print(Y_pred[:10])

# %%
#observe y_test and y_predict
print(Y_pred[:20])
print(Y_test[:20])

# %%
"""
### Confusion Matrix
"""

# %%
#computing the confusion matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test, Y_pred)
print(cm)

# TP FP
# FN TN

# %%
#calculating precision
precision=cm[0][0]/(cm[0][0]+cm[0][1])
print(precision)

#precision=TP/(TP+FP)

# %%
#calculating accuracy
total=cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]
accuracy=(cm[0][0]+cm[1][1])/total
print(accuracy*100, "%")

#accuracy=(TP+TN)/Total

# %%
#calculating error rate
error=(cm[0][1]+cm[1][0])/total
print(error*100, "%")

#error rate=(FP+FN)/Total

# %%
#calculating recall
recall=cm[0][0]/(cm[0][0]+cm[1][0])
print(recall)

#recall=TP/(TP+FN)

# %%
#auc-roc curve
from sklearn import metrics
auc = metrics.roc_auc_score(Y_test, Y_pred)
false_positive_rate, true_positive_rate, thresolds = metrics.roc_curve(Y_test, Y_pred)

plt.figure(figsize=(3, 3), dpi=100)

plt.xlim([0, 1])
plt.ylim([0, 1])
plt.title("AUC & ROC Curve")
plt.plot(false_positive_rate, true_positive_rate, 'g')
plt.fill_between(false_positive_rate, true_positive_rate, facecolor='lightgreen', alpha=0.7)
plt.text(0.95, 0.05, 'AUC = %0.4f' % auc, ha='right', fontsize=12, weight='bold', color='blue')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()