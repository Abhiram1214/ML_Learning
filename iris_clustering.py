# -*- coding: utf-8 -*-
"""
Created on Fri May  3 21:26:00 2019

@author: abhiram_ch_v_n_s
"""

from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd


# Loading dataset
iris_df = datasets.load_iris()

print(dir(iris_df))

print(iris_df.DESCR)

x_axis =  iris_df.data[:,0]
y_axis = iris_df.data[:,2]

label = {0: 'red', 1: 'blue', 2: 'green'}

flower = ['setosa', 'versicolor', 'virginica']

plt.scatter(x_axis, y_axis, c = iris_df.target, label=flower)
plt.legend()

#-----------

model = KMeans(n_clusters=3)
model.fit(iris_df.data)


predict = model.predict([[4.9,3,1.4,0.2]])

all_predictions = model.predict(iris_df.data)

col = list(iris_df.feature_names)

df = pd.DataFrame(iris_df.data, columns=[col])
df['predictions'] = all_predictions


color = ['r.','g.','b.']
cluster_centers = model.cluster_centers_

plt.scatter(df['sepal length (cm)'], df['petal length (cm)'], color='r')
plt.scatter(df['sepal width (cm)'], df['petal width (cm)'], color='g')
plt.scatter(cluster_centers[:,0], cluster_centers[:,2])
plt.scatter(cluster_centers[:,1], cluster_centers[:,3])
#plt.plot(model.labels_)

'''
plt.scatter(df['sepal length (cm)'], df['petal length (cm)'])


plt.scatter(iris_df.data[:,0], iris_df.data[:,2])
plt.scatter(model.cluster_centers_[0][0], model.cluster_centers_[0][2], marker='+')
'''


#---------------------------------------------



#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('iris.csv')

dataset.head()

x = dataset.iloc[:,[1,2,3,4]].values

from sklearn.cluster import KMeans
wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    

plt.plot(range(1,11), wcss)    
    
#number of clusters = 3

kmeans = KMeans(n_clusters=3)
y_kmeans = kmeans.fit_predict(x)




plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 100, c = 'yellow', label = 'Centroids')

plt.legend()




















