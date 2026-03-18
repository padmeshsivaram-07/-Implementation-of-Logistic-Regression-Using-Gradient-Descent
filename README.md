# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Start the program.
2.Data preprocessing:
3.Cleanse data,handle missing values,encode categorical variables.
4.Model Training:Fit logistic regression model on preprocessed data.
5.Model Evaluation:Assess model performance using metrics like accuracyprecisioon,recall.
6.Prediction: Predict placement status for new student data using trained model.
7.End the program. 

## Program:
```
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split

data = pd.read_csv("C:/Users/acer/Downloads/Placement_Data.csv")
data.drop("sl_no", axis=1, inplace=True)

data['status'] = data['status'].map({'Placed': 1, 'Not Placed': 0})
data = pd.get_dummies(data, drop_first=True)

X = data.drop('status', axis=1).values
y = data['status'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

X_train = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)
X_test  = (X_test  - X_test.mean(axis=0))  / X_test.std(axis=0)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
weights = np.zeros(X_train.shape[1])
bias = 0
learning_rate = 0.1     
epochs = 3000

for _ in range(epochs):
    linear = np.dot(X_train, weights) + bias
    y_pred = sigmoid(linear)

    dw = (1 / len(y_train)) * np.dot(X_train.T, (y_pred - y_train))
    db = (1 / len(y_train)) * np.sum(y_pred - y_train)

    weights -= learning_rate * dw
    bias -= learning_rate * db

def predict(X):
    linear = np.dot(X, weights) + bias
    return np.where(sigmoid(linear) >= 0.5, 1, 0)

y_predicted = predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_predicted) * 100, "%")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_predicted))

print("\nClassification Report:")
print(classification_report(y_test, y_predicted))
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Padmesh Sivaram R
RegisterNumber:  25012017
*/
```

## Output:
<img width="821" height="395" alt="image" src="https://github.com/user-attachments/assets/33282f5e-c0db-410f-8124-188c0f6024c6" />



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

