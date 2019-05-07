# -*- coding: utf-8 -*-
"""
Created on Wed May  8 03:42:03 2019

@author: abhiram_ch_v_n_s
"""

import pandas as pd

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

from sklearn.datasets.samples_generator import make_blobs


X, y = make_blobs(n_samples=600, centers=5,
                  cluster_std=0.6, random_state=42)

plt.scatter(X[:,0], X[:,1],s=10)

from scipy.cluster.hierarchy import ward, dendrogram, linkage
np.set_printoptions(precison=4, supress=True)


distance = linkage(X, 'ward')

#Dendogram

plt.figure(figsize=(25,10))
plt.title("hierarchical clustering")
plt.xlabel("index")
plt.ylabel("Ward's distance")

dendrogram(distance, orientation='left',
           leaf_rotation=90,
           leaf_font_size=6)


#K-Means clustering

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4)
kmeans.fit(X)

y_kmeans = kmeans.predict(X)

plt.scatter(X[:,0], X[:,1],c=y_kmeans,s=10,cmap='inferno')

centers = kmeans.cluster_centers_
plt.scatter(centers[:,0], centers[:,1], c='green')

from mlxtend.plotting import plot_decission_regions
plot_decission_regions(X,y, clf=kmeans);