# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 16:19:23 2024

@author: Admin
"""

'''
2. Divide the diabetes data into train and test datasets
 and build a  Decision Tree model with 
 Outcome as the output variable. 
Business Contraints
Interpretability: The Decision Tree model needs to be interpretable so that stakeholders can understand the factors contributing to classifying individuals as risky or good.

Scalability: The model should be scalable to handle a large volume of data as the company expands its operations or encounters more data.

Accuracy: While interpretability is important, the model should also be accurate in predicting whether an individual is risky or good based on their taxable income.

Cost of Misclassification: Minimize the misclassification of individuals, especially those who are actually risky but classified as good. False negatives (misclassifying risky individuals as good) can be costly for the company in terms of fraud.

Maximize:
Maximize the accuracy of the Decision Tree model. 
Higher accuracy ensures that the model can effectively
 distinguish between risky and good individuals based on 
 their taxable income.
 
Minimize:
 Minimize the cost associated with misclassifications,
 particularly false negatives. Misclassifying risky individuals
 as good may lead to potential fraud-related losses for the
 company. Thus, minimizing false negatives is crucial to 
 mitigate such risks and associated costs.
 '''


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("Diabetes.csv")

data.isnull().sum()
data.dropna()
data.columns


#Converting into binary
lb = LabelEncoder()
data[" Number of times pregnant"] = lb.fit_transform(data[" Number of times pregnant"])
data[" Plasma glucose concentration"] = lb.fit_transform(data[" Plasma glucose concentration"])
data["Diastolic blood pressure"] = lb.fit_transform(data["Diastolic blood pressure"])
data[" Triceps skin fold thickness"] = lb.fit_transform(data[" Triceps skin fold thickness"])
data[" 2-Hour serum insulin"] = lb.fit_transform(data[" 2-Hour serum insulin"])
data[" Body mass index"]=lb.fit_transform(data[" Body mass index"])
data[" Diabetes pedigree function"]=lb.fit_transform(data[" Diabetes pedigree function"])
data[" Age (years)"]=lb.fit_transform(data[" Age (years)"])
data['default'].unique()
data['default'].value_counts()
colnames=list(data.columns)

predictors=colnames[:15]
target=colnames[15]

#Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
train,test=train_test_split(data,test_size=0.3)

from sklearn.tree import DecisionTreeClassifier as DT
#help(DT)
model=DT(criterion='entropy')
model.fit(train[predictors],train[target])
preds=model.predict(test[predictors])
preds
pd.crosstab(test[target],preds,rownames=['Actual'],colnames=['predictions'])
np.mean(preds==test[target])

#Now let us check accuracy on training dataset
preds_train=model.predict(train[predictors])
pd.crosstab(train[target],preds_train,rownames=['Actual'],colnames=['predictions'])
np.mean(preds_train==train[target])