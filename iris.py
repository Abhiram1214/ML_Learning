#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 07:35:51 2019

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
from sklearn.cluster import KMeans


import statsmodels.api as sm
import sklearn
import scipy as stats



iris_df = pd.read_csv('Iris.csv')
iris_df = iris_df.drop('Id', axis=1)

X = iris_df.drop('Species',axis=1)
y = iris_df['Species']

#y is categorical

encoder = LabelEncoder()
y = encoder.fit_transform(y)

#model 
model = KMeans(n_clusters=3)
model.fit(X)

y_kmeans = model.predict(X)
centers = model.cluster_centers_

plt.scatter(X.iloc[:,0], X.iloc[:,1], c=y_kmeans)
plt.scatter(X.iloc[:,2], X.iloc[:,3], c=y_kmeans)
plt.scatter(centers[2,:], c='yellow')
plt.scatter(centers[:,0], centers[:,1])
plt.scatter(centers[:,2], centers[:,3])








