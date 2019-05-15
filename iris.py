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
from mpl_toolkits.mplot3d import Axes3D

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
iris_df['clusters'] = model.labels_

y_kmeans = model.predict(X)
centers = model.cluster_centers_






plt.scatter(X.iloc[:,0], X.iloc[:,1], c=y_kmeans)
plt.scatter(X.iloc[:,1], X.iloc[:,3], c=y_kmeans)



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = iris_df.iloc[:,0]
y = iris_df.iloc[:,1]


plt.scatter(iris_df.iloc[:,0], iris_df.iloc[:,1], c=y_kmeans)
plt.scatter(centers[:,0], centers[:,1])

plt.figure('K-Means on Iris Dataset', figsize=(7,7))
ax = plt.axes(projection = '3d')
ax.scatter(X.iloc[:,3],X.iloc[:,0],X.iloc[:,2], c=y , cmap='Set2', s=50)
ax.scatter(centers[0,3],centers[0,0],centers[0,2] ,c='r', s=50, label='centroid')
ax.scatter(centers[1,3],centers[1,0],centers[1,2] ,c='r', s=50)
ax.scatter(centers[2,3],centers[2,0],centers[2,2] ,c='r', s=50)





plt.scatter(X.iloc[:,3],X.iloc[:,0], c=y , cmap='Set2', s=50)
plt.scatter(centers[:,3], centers[:,0],c='r')





