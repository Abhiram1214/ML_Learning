# -*- coding: utf-8 -*-
"""
Created on Sat May  4 04:13:53 2019

@author: abhiram_ch_v_n_s
"""

import pandas as pd

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

from sklearn.preprocessing import MinMaxScaler

# machine learning
from sklearn.linear_model import LogisticRegression

heart_df = pd.read_csv('framingham.csv')

heart_df.info()

#-------correlation----------
corr = heart_df.corr()
sns.heatmap(corr, annot=True, cmap="YlGnBu")

sorted_cor = (corr['TenYearCHD']).sort_values(ascending=False)

#['age', 'sysBP', 'prevalentHyp', 'diaBP', 'education', 'glucose','currentSmoker']



#EDA
heart_df.isnull().sum()

heart_df[heart_df['education'].isnull()]

heart_df['heartRate'].mean()

heart_df['education'][np.isnan(heart_df['education'])] = 2
heart_df['cigsPerDay'][np.isnan(heart_df['cigsPerDay'])] = 9 
heart_df['BPMeds'][np.isnan(heart_df['BPMeds'])] = 0 
heart_df['totChol'][np.isnan(heart_df['totChol'])] = 237
heart_df['BPMeds'][np.isnan(heart_df['BPMeds'])] = 0 
heart_df['BMI'][np.isnan(heart_df['BMI'])] = 26
heart_df['glucose'][np.isnan(heart_df['glucose'])] = 82 
heart_df['heartRate'][np.isnan(heart_df['heartRate'])] = 76


heart_df.isnull().sum()



#Logistic Regression
X = heart_df.drop('TenYearCHD', axis=1)
y = heart_df['TenYearCHD']

model = LogisticRegression()
model.fit(X,y)

prediction = model.predict(X)

model.score(X,y)
#0.8544811320754717


#after scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)



model = LogisticRegression()
model.fit(X_scaled,y)

prediction = model.predict(X_scaled)

model.score(X_scaled,y)
#0.8549528301886793




#Feature selection
#-----------------------------------
import statsmodels.api as sm
from scipy import stats
import sklearn

X2 = sm.add_constant(X)
est = sm.Logit(y, X2)
est2 = est.fit()

print(est2.summary())


X_new = heart_df[['age','male','cigsPerDay','totChol','sysBP','glucose','prevalentStroke']]
Y_new = heart_df['TenYearCHD']


from sklearn.model_selection import train_test_split


x_train, x_test, y_train, y_test = train_test_split(X_new, Y_new, test_size=.40, random_state=5)


log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)


y_pred = log_reg.predict(x_test)

sklearn.metrics.accuracy_score(y_test,y_pred)
#0.8555424528301887


#---------lETS SCALE THE X_NEW

x_scaler = MinMaxScaler()
X_new_scaled = x_scaler.fit_transform(X_new)




X_new = heart_df[['age','male','cigsPerDay','totChol','sysBP','glucose']]
Y_new = heart_df['TenYearCHD']


from sklearn.model_selection import train_test_split


x_train, x_test, y_train, y_test = train_test_split(X_new_scaled, Y_new, test_size=.40, random_state=5)


log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)


y_pred = log_reg.predict(x_test)

sklearn.metrics.accuracy_score(y_test, y_pred)
#0.8567216981132075


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

TP = cm[0,0]
TN = cm[1,1]
FN = cm[1,0]
FP = cm[0,1]

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


y_pred_proba = log_reg.predict_proba(x_test)[:,:]
y_pred_prob_df = pd.DataFrame(data=y_pred_proba, columns=['Prob of no heart disease (0)','Prob of Heart Disease (1)'])
y_pred_prob_df.head()










