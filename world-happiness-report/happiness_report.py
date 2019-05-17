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
from sklearn.linear_model import LogisticRegression
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

world_df = world_df.dropna(axis=1)

X = world_df.drop(['Happiness Rank', 'Happiness Score', 'Country', 'Region','Health (Life Expectancy)'], axis=1)
y = world_df['Happiness Score']




#y is continuous 

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)




#------Pvalue
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler() 
scaler.fit(y)
y_scaled = scaler.fit_transform(y)



import statsmodels.api as sm

y = y.reshape(1,-1)
est = sm.Logit(y_encoded, X)
est2 = est.fit()
print(est2.summary())




#For 2015 data
X_2015_data = data_2015.iloc[:,4:] 
y_2015_data = data_2015.iloc[:,3]


est = sm.OLS(y_2015_data, X_2015_data)
est2 = est.fit()
est2.summary()
#except standard error, all are in corelation with the column score

data_2015_corr = world_df.corr()
sns.heatmap(data_2015_corr, annot=True)


#Linear Regression

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=3)

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




lm.predict([[0.03729, 1.42727, 1.12575, 0.80925, 0.64157, 0.38583]])





