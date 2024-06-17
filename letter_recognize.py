# -*- coding: utf-8 -*-
"""「Letter Recognize

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1SPLJ_CTM1Nogdrw1hI2kyRQMQhx0EJGA

#### Importing libraries
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn import linear_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

"""#### Load the dataset and Split features and target variable"""

# Load the dataset
data = pd.read_csv('/content/finaln.csv')
# Split features and target variable
X = data
y = data['0']

for i in range(1):
    dd = data[data['0']==i].iloc[1]
    dd1 = data[data['0']==i].iloc[2]
    x = dd.values
    x = x.reshape((28, 28))
    x1 = dd1.values
    x1 = x1.reshape((28, 28))
    plt.imshow(x, cmap='binary')
    plt.show()
    plt.imshow(x1, cmap='binary')
    plt.show()

data.head()

data.describe()

data.info()

"""#### Splitting the data into training and testing sets"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.shape,X_test.shape

y_train.value_counts()

X_train.values

y_train.values

"""#### Create an SVM classifier"""

svm_model = svm.SVC(kernel='linear')
svm_model.fit(X_train.values,y_train.values)
#svm_predictions = svm_model.predict(X_test)
#svm_accuracy = accuracy_score(y_test, svm_predictions)
svmtrain_score = svm_model.score(X_train.values, y_train.values)
svmtest_score = svm_model.score(X_test.values, y_test.values)

print(svmtrain_score)
print(svmtest_score)

d2d=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,7,38,72,124,166,209,218,218,222,211,174,138,84,46,11,0,0,0,0,0,0,0,0,0,0,0,0,0,116,203,246,255,255,255,255,255,255,255,255,255,250,214,154,12,0,0,0,0,0,0,0,0,0,0,0,0,223,247,207,153,110,69,44,44,44,63,95,140,207,255,247,15,0,0,0,0,0,0,0,0,0,0,0,0,100,117,16,0,0,0,0,0,0,0,0,0,107,255,190,0,0,0,0,0,0,0,0,0,0,0,0,0,1,2,0,0,0,0,0,0,0,0,0,15,200,248,98,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,110,244,180,25,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,35,196,243,100,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,124,249,190,16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,28,230,243,78,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,132,255,177,17,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,71,227,222,62,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,14,164,253,144,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,82,247,229,28,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,24,202,248,114,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,115,246,195,35,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,62,223,237,79,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,171,254,165,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,77,250,252,140,81,81,81,81,81,79,53,44,44,44,42,22,0,0,0,0,0,0,0,0,0,0,0,9,168,255,255,255,255,255,255,255,255,255,255,255,255,255,251,181,0,0,0,0,0,0,0,0,0,0,0,45,150,107,136,167,181,181,181,184,213,218,218,218,218,218,227,164,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
print(chr((svm_model.predict([d2d])+65)[0]))
#print(svm_model.predict([d2d]))
d1d=d2d
#d1d.append(0)
d1d=pd.Series(d1d)
#xs=d1d[1:].values
xs=d1d.values
xs=xs.reshape((28, 28))
plt.imshow(xs, cmap='gray')
plt.axis('off')
plt.show()

d2d=[20, 255, 255, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 250, 253, 255, 255, 255, 255, 255, 253, 252, 254, 255, 255, 255, 255, 255, 254, 250, 254, 255, 255, 255, 255, 255, 252, 252, 254, 255, 251, 251, 255, 255, 255, 255, 255, 255, 252, 252, 254, 255, 255, 255, 255, 254, 250, 252, 255, 255, 255, 255, 255, 254, 252, 252, 254, 254, 252, 255, 255, 255, 255, 255, 255, 255, 254, 254, 255, 255, 255, 255, 255, 253, 252, 255, 255, 255, 255, 255, 255, 255, 253, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 249, 157, 137, 140, 139, 128, 178, 219, 141, 139, 136, 131, 132, 137, 140, 140, 137, 179, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 234, 42, 21, 23, 22, 82, 192, 248, 205, 196, 185, 156, 105, 41, 15, 23, 19, 81, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 234, 48, 32, 28, 110, 255, 255, 255, 255, 255, 255, 255, 255, 229, 123, 30, 26, 85, 255, 255, 255, 255, 255, 255, 255, 254, 255, 255, 234, 48, 30, 30, 73, 218, 255, 255, 255, 255, 255, 255, 255, 255, 255, 126, 20, 85, 255, 255, 255, 254, 254, 255, 253, 252, 254, 255, 234, 48, 30, 33, 23, 133, 255, 255, 255, 208, 136, 149, 186, 246, 255, 236, 44, 83, 255, 255, 254, 250, 254, 255, 252, 252, 254, 255, 234, 48, 30, 34, 38, 154, 255, 255, 245, 66, 13, 12, 85, 243, 255, 247, 53, 81, 255, 254, 250, 253, 255, 255, 254, 254, 255, 255, 234, 48, 30, 35, 39, 206, 255, 255, 214, 46, 73, 122, 229, 255, 255, 211, 29, 84, 255, 253, 253, 255, 255, 255, 255, 255, 255, 255, 234, 48, 32, 28, 70, 250, 255, 255, 241, 227, 253, 255, 255, 255, 229, 90, 20, 85, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 234, 50, 31, 24, 129, 255, 255, 255, 255, 255, 255, 255, 255, 181, 42, 27, 29, 86, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 234, 50, 30, 29, 194, 255, 255, 255, 255, 255, 255, 255, 255, 234, 129, 28, 28, 87, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 234, 48, 29, 53, 239, 255, 255, 232, 171, 146, 153, 174, 228, 255, 255, 137, 21, 85, 255, 255, 255, 255, 255, 255, 255, 254, 253, 255, 234, 48, 23, 102, 255, 255, 247, 79, 20, 23, 19, 14, 145, 255, 255, 243, 49, 82, 255, 255, 255, 254, 255, 255, 254, 250, 253, 255, 234, 48, 21, 149, 255, 255, 183, 19, 23, 31, 65, 147, 246, 250, 255, 245, 50, 82, 255, 255, 252, 252, 254, 254, 250, 252, 255, 255, 234, 48, 24, 185, 255, 253, 121, 87, 145, 201, 247, 255, 255, 255, 254, 169, 24, 85, 255, 255, 252, 252, 255, 254, 253, 255, 255, 255, 234, 42, 21, 206, 255, 253, 241, 255, 255, 255, 255, 247, 223, 172, 88, 32, 26, 85, 255, 255, 254, 255, 255, 255, 255, 255, 255, 255, 231, 97, 139, 239, 255, 255, 255, 247, 221, 167, 108, 67, 42, 26, 26, 34, 27, 84, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 231, 112, 194, 249, 246, 159, 87, 57, 37, 22, 20, 23, 26, 27, 28, 28, 23, 86, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 253, 196, 185, 244, 244, 183, 182, 186, 189, 191, 190, 190, 190, 190, 190, 190, 188, 215, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 254, 255, 255, 255, 255, 255, 255, 255, 254, 253, 255, 255, 255, 255, 255, 254, 254, 255, 255, 255, 255, 255, 255, 255, 252, 254, 255, 253, 252, 253, 255, 255, 255, 255, 255, 254, 250, 254, 255, 255, 255, 255, 255, 252, 252, 254, 255, 255, 255, 255, 255, 251, 250, 255, 255, 252, 252, 254, 255, 255, 255, 255, 254, 250, 253, 255, 255, 255, 255, 255, 254, 252, 253, 255, 255, 255, 255, 255, 252, 250, 255, 255, 255, 254, 254, 255, 255, 255, 255, 255, 254, 253, 255, 255, 255, 255, 255, 255, 255, 254, 255, 255, 255, 255, 255, 255, 253, 255, 255, 255]
print(chr((svm_model.predict([d2d])+65)[0]))
d1d=d2d
d1d.append(0)
d1d=pd.Series(d1d)
xs=d1d[1:].values
xs=xs.reshape((28, 28))
plt.imshow(xs, cmap='gray')
plt.axis('off')
plt.show()

"""#### Create a KNN classifier"""

knn_model = KNeighborsClassifier()
knn_model.fit(X_train.values, y_train.values)
#knn_predictions = knn_model.predict(X_test)
#knn_accuracy = accuracy_score(y_test, knn_predictions)
knntrain_score = knn_model.score(X_train.values, y_train.values)
knntest_score = knn_model.score(X_test.values, y_test.values)

print(knntrain_score)
print(knntest_score)

"""#### Create a Decision Tree classifier"""

dt_model = DecisionTreeClassifier()
dt_model.fit(X_train.values, y_train.values)
#dt_predictions = dt_model.predict(X_test)
#dt_accuracy = accuracy_score(y_test, dt_predictions)
dttrain_score = dt_model.score(X_train.values, y_train.values)
dttest_score = dt_model.score(X_test.values, y_test.values)

print(dttrain_score)
print(dttest_score)

"""#### Create a Random Forest classifier"""

rf_model = RandomForestClassifier(n_estimators=150)
rf_model.fit(X_train.values, y_train.values)
#rf_predictions = rf_model.predict(X_test)
#rf_accuracy = accuracy_score(y_test, rf_predictions)
rftrain_score = rf_model.score(X_train.values, y_train.values)
rftest_score = rf_model.score(X_test.values, y_test.values)

print(rftrain_score)
print(rftest_score)

"""#### Create a Naive Bayes classifier"""

nb_model = GaussianNB()
nb_model.fit(X_train.values, y_train.values)
#nb_predictions = nb_model.predict(X_test)
#nb_accuracy = accuracy_score(y_test, nb_predictions)
nbtrain_score = nb_model.score(X_train.values, y_train.values)
nbtest_score = nb_model.score(X_test.values, y_test.values)

print(nbtrain_score)
print(nbtest_score)

mlp_model = MLPClassifier()
mlp_model.fit(X_train, y_train)
#mlp_predictions = mlp_model.predict(X_test)
#mlp_accuracy = accuracy_score(y_test, mlp_predictions)
mlptrain_score = mlp_model.score(X_train, y_train)
mlptest_score = mlp_model.score(X_test, y_test)

print(mlptrain_score)
print(mlptest_score)

lr_model = LogisticRegression()
lr_model.fit(X_train.values, y_train.values)
#lr_predictions = lr_model.predict(X_test)
#lr_accuracy = accuracy_score(y_test, lr_predictions)
lrtrain_score = mlp_model.score(X_train.values, y_train.values)
lrtest_score = mlp_model.score(X_test.values, y_test.values)

print(lrtrain_score)
print(lrtest_score)

ridge_model = Ridge()
ridge_model.fit(X_train, y_train)
ritrain_score = ridge_model.score(X_train.values, y_train.values)
ritest_score = ridge_model.score(X_test.values, y_test.values)

print(ritrain_score)
print(ritest_score)

lasso_model = Lasso().fit(X_train.values, y_train.values)
latrain_score = lasso_model.score(X_train.values, y_train.values)
latest_score = lasso_model.score(X_test.values, y_test.values)

print(latrain_score)
print(latest_score)

"""#### Compare the accuracies of the models"""

performance_metrics = pd.DataFrame({
    'Algorithm': ['SVM','KNN','DT','RF','NB','MLP','LR','RI','LA'],
    'TrainSC': [svmtrain_score, knntrain_score, dttrain_score, rftrain_score, nbtrain_score, mlptrain_score, lrtrain_score,ritrain_score,latrain_score],
    'TestSC': [svmtest_score, knntest_score, dttest_score, rftest_score, nbtest_score, mlptest_score, lrtest_score,ritest_score,latest_score],
})

print(performance_metrics)

ax = performance_metrics.plot.bar(x='Algorithm',rot=0)

"""## Data visualization

#### Class distribution
"""

plt.figure(figsize=(8, 6))
sns.countplot(x=y)
plt.xlabel('Letter')
plt.ylabel('Count')
plt.title('Class Distribution')
plt.show()

"""#### Plotting histograms for each feature"""

plt.figure(figsize=(12, 8))
for i, column in enumerate(X.columns):
    plt.subplot(3, 7, i + 1)
    plt.hist(X[column], bins=20, edgecolor='black')
    plt.xlabel(column)
    plt.ylabel('Count')
plt.tight_layout()
plt.show()

"""#### Plotting each letter"""

unique_letters = sorted(data[0].unique())

plt.figure(figsize=(16, 10))
for i, letter in enumerate(unique_letters):
    plt.subplot(4, 7, i + 1)
    letter_data = data[data[0] == letter].drop(columns=0)
    letter_data = letter_data.reset_index(drop=True)
    for j in range(len(letter_data)):
        plt.plot(range(1, len(letter_data.columns) + 1), letter_data.iloc[j, :], marker='o', linewidth=0.5)
    plt.xlabel('Feature')
    plt.ylabel('Value')
    plt.title(f'Letter: {letter}')
plt.tight_layout()
plt.show()

"""#### Confusion matrix for SVM"""

plt.figure(figsize=(8, 6))
svm_cm = pd.crosstab(y_test, svm_predictions, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(svm_cm, annot=True, cmap='Blues')
plt.title('Confusion Matrix - SVM')
plt.show()

"""#### Confusion matrix for KNN"""

plt.figure(figsize=(8, 6))
knn_cm = pd.crosstab(y_test, knn_predictions, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(knn_cm, annot=True, cmap='Blues')
plt.title('Confusion Matrix - KNN')
plt.show()

"""#### Confusion matrix for Decision Tree"""

plt.figure(figsize=(8, 6))
dt_cm = pd.crosstab(y_test, dt_predictions, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(dt_cm, annot=True, cmap='Blues')
plt.title('Confusion Matrix - Decision Tree')
plt.show()

"""#### Confusion matrix for Random Forest"""

plt.figure(figsize=(8, 6))
rf_cm = pd.crosstab(y_test, rf_predictions, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(rf_cm, annot=True, cmap='Blues')
plt.title('Confusion Matrix - Random Forest')
plt.show()

"""#### Confusion matrix for Naive Bayes"""

plt.figure(figsize=(8, 6))
nb_cm = pd.crosstab(y_test, nb_predictions, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(nb_cm, annot=True, cmap='Blues')
plt.title('Confusion Matrix - Naive Bayes')
plt.show()

"""#### Plotting a few input images"""

n_examples = 8
example_indices = np.random.choice(range(len(X_train)), size=n_examples, replace=False)

for i, idx in enumerate(example_indices):
    plt.subplot(2, 4, i+1)
    example_image = X_train.iloc[idx, :].values.reshape(4, 4)
    plt.imshow(example_image, cmap='binary')
    plt.title(f"Letter: {y_train.iloc[idx]}")
    plt.axis('off')
plt.tight_layout()
plt.show()