#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 09:16:13 2019

@author: abhiram
"""


import pandas as pd

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


import statsmodels.api as sm
import sklearn
import scipy as stats


data_2015 = pd.read_csv('2015.csv')
data_2016 = pd.read_csv('2016.csv')
data_2017 = pd.read_csv('2017.csv')



world_df = pd.DataFrame()
world_df = world_df.append(data_2015)
world_df = world_df.append(data_2016)

data_2015.isnull().sum()
data_2015_dummy = data_2015.copy()

'''
# see if there is any correlation between country and region with the happiness score

## create dummy variables for Person column, & drop Male as it has the lowest average of survived passengers

data_2015_dummy_h = pd.get_dummies(data_2015_dummy[['Country','Region']])
data_2015_dummy = data_2015_dummy.join(data_2015_dummy_h)
data_2015_dummy = data_2015_dummy.drop(['Country','Region'],axis=1)
'''


#For 2015 data
X_2015_data = data_2015.iloc[:,4:] 
y_2015_data = data_2015.iloc[:,3]


est = sm.OLS(y_2015_data, X_2015_data)
est2 = est.fit()
est2.summary()
#except standard error, all are in corelation with the column score

data_2015_corr = data_2015.corr()
sns.heatmap(data_2015_corr, annot=True)


#Linear Regression

x_train, x_test, y_train, y_test = train_test_split(X_2015_data, y_2015_data, test_size=0.40, random_state=3)

lm = LinearRegression()
lm.fit(x_train, y_train)

y_pred = lm.predict(x_test)



#Evaluation metrics

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_pred, y_test)
print(mse)

rmse = np.sqrt(mse)
print(rmse)

#predict for 2016

X_2016_data = data_2016.iloc[:,4:] 
y_2016_data = data_2016.iloc[:,3]




lm.predict([[0.03729, 1.42727, 1.12575, 0.80925, 0.64157, 0.38583, 0.26428,	2.24743]])


#--------------------------------

world_df.isnull().sum()

world_df = world_df.dropna(axis=1)



from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder


X = world_df.drop(['Country', 'Happiness Rank', 'Happiness Score', 'Region', 'Dystopia Residual'], axis=1)
y = world_df['Happiness Score']


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.40, random_state=42)

lm = LinearRegression()

lm.fit(X_train, y_train)
y_pred = lm.predict(X_test)

#MSE 
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(rmse)
#0.5399329163493787

lm.predict([[1.39651, 1.34951, 0.6657, 0.29678, 0.94143, 0.41978]])


sns.regplot(world_df['Family'], y, data=world_df)
#world_df.groupby(by=['Happiness Score','Freedom'])








#logistic regression
#y is a continuous value... for LogReg the result should be 0 or 1
# ans is zero

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)


log_pred = log_reg.predict(X_test)

sklearn.metrics.accuracy_score(y_test,log_pred)
#0.9736842105263158


#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, log_pred)




