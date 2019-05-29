#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 12:25:03 2019

@author: abhiram
"""

import numpy as np
import pandas as pd
import keras

dataset = pd.read_csv('Iris.csv')

X = dataset.iloc[:,1:5].values
y = dataset.iloc[:, 5].values

y = pd.get_dummies(dataset.Species)


# since the output is categorical, we can use softmax function

from keras.models import Sequential#sequential module to initialize ANN
from keras.layers import Dense#Dense module to build the layers of the ANN

classifer = Sequential()
classifer.add(Dense(4, input_dim=4, kernel_initializer='uniform', activation='relu'))
classifer.add(Dense(3, kernel_initializer='uniform', activation='sigmoid'))

classifer.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#classifer.fit(X,y, nb_epoch=110)
classifer.fit(X,y, epochs=125)

y_pred = classifer.predict_classes(X)


from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

target = dataset.iloc[:, 5].values
labelencoder= LabelEncoder()
target = labelencoder.fit_transform(target)

cm = confusion_matrix(y_pred, target)



