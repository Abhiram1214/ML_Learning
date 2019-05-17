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


plt.figure('K-Means on Iris Dataset', figsize=(7,7))
ax = plt.axes(projection = '3d')
ax.scatter(X.iloc[:,3],X.iloc[:,0],X.iloc[:,2], c=y , cmap='Set2', s=50)
ax.scatter(centers[0,3],centers[0,0],centers[0,2] ,c='r', s=50, label='centroid')
ax.scatter(centers[1,3],centers[1,0],centers[1,2] ,c='r', s=50)
ax.scatter(centers[2,3],centers[2,0],centers[2,2] ,c='r', s=50)





plt.scatter(X.iloc[:,0],X.iloc[:,2], c=y_kmeans, cmap='Set2', s=50)
plt.scatter(centers[:,0], centers[:,2],c='r')


#------------KNN

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=3)

knn =  KNeighborsClassifier()

knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)
accuracy = knn.score(x_test, y_test)

knn.predict([[3, 5.2, 4.7, 0.4]])

a = knn.kneighbors_graph(X)
a.toarray()



plt.figure()

ax=plt.scatter(X.iloc[:, 0], X.iloc[:, 1],c=y)

#------------Logistic

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=3)


log = LogisticRegression()
log.fit(x_train, y_train)
log_pred = log.predict(x_test)

log.score(log_pred, y_test)
log.predict_proba(X)



















