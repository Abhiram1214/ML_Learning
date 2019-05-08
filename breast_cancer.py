# -*- coding: utf-8 -*-
"""
Created on Tue May  7 23:50:09 2019

@author: abhiram_ch_v_n_s
"""


import pandas as pd

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder



import statsmodels.api as sm
import sklearn
import scipy as stats



# machine learning
from sklearn.linear_model import LogisticRegression


cancer_df = pd.read_csv('breast_cancer.csv')
cancer_df = cancer_df.drop('Unnamed: 32', axis=1)

cancer_df.info()

cancer_df.isnull().sum()
#no null values.


y = cancer_df['diagnosis']
encoder_y = LabelEncoder()
y= encoder_y.fit_transform(y)

X = cancer_df.drop(['id', 'diagnosis'], axis=1)

#calculate p-value
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)


#Logistic Regression: First version----------

model = LogisticRegression()
model.fit(X,y)

predictions = model.predict(X)
model.score(X,y)
#0.9595782073813708


#----for scaled values---------
model_scaled = LogisticRegression()
model_scaled.fit(X_scaled,y)

predictions = model_scaled.predict(X_scaled)
model_scaled.score(X_scaled,y)
#0.9666080843585237



from sklearn.model_selection import train_test_split


x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=.40, random_state=5)


log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)


y_pred = log_reg.predict(x_test)

sklearn.metrics.accuracy_score(y_test,y_pred)
#0.9736842105263158


#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]


#accuracy
#accuracy = correct predictions/total observations


sensitivity=TP/float(TP+FN)
specificity=TN/float(TN+FP)

print('The acuuracy of the model = TP+TN/(TP+TN+FP+FN) = ',(TP+TN)/float(TP+TN+FP+FN),'\n',

'The Missclassification = 1-Accuracy = ',1-((TP+TN)/float(TP+TN+FP+FN)),'\n',

'Sensitivity or True Positive Rate = TP/(TP+FN) = ',TP/float(TP+FN),'\n',

'Specificity or True Negative Rate = TN/(TN+FP) = ',TN/float(TN+FP),'\n',

'Positive Predictive value = TP/(TP+FP) = ',TP/float(TP+FP),'\n',

'Negative predictive Value = TN/(TN+FN) = ',TN/float(TN+FN),'\n',

'Positive Likelihood Ratio = Sensitivity/(1-Specificity) = ',sensitivity/(1-specificity),'\n',

'Negative likelihood Ratio = (1-Sensitivity)/Specificity = ',(1-sensitivity)/specificity)



from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

'''

y_pred_proba = log_reg.predict_proba(x_test)[:,:]
y_pred_prob_df = pd.DataFrame(data=y_pred_proba, columns=['Prob of a bengin tumor','Prob of malignant tumor'])
y_pred_prob_df.head()
'''


#--------------------------------

#clustring
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5)
kmeans.fit(X)

pred = kmeans.predict(X)

cluster = kmeans.cluster_centers_
pred != kmeans.labels_

plt.scatter(X.iloc[:,0], X.iloc[:,1], c=pred, s=20)

plt.scatter(cluster[:,0], cluster[:,1])




