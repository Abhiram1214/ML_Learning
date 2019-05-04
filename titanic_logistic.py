# -*- coding: utf-8 -*-
"""
Created on Sat May  4 01:15:13 2019

@author: abhiram_ch_v_n_s
"""

import pandas as pd
import pandas as pd

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')


# machine learning
from sklearn.linear_model import LogisticRegression

titanic_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

titanic_df.head()

titanic_df.info()


# drop unnecessary columns, these columns won't be useful in analysis and prediction
titanic_df = titanic_df.drop(['PassengerId','Name','Ticket'], axis=1)
test_df    = test_df.drop(['Name','Ticket'], axis=1)


titanic_df["Embarked"].value_counts()
titanic_df.isnull().sum()

titanic_df["Embarked"] = titanic_df['Embarked'].fillna('s')

#----------dummy variable 

#embark_dummies_titanic  = pd.get_dummies(titanic_df['Embarked'])
#embark_dummies_titanic.drop(['S'], axis=1, inplace=True)
#titanic_df = titanic_df.join(embark_dummies_titanic)


titanic_df.drop(['Embarked'], axis=1,inplace=True)
test_df.drop(['Embarked'], axis=1,inplace=True)


        
means = test_df['Age'].mean()
titanic_mean = titanic_df['Age'].mean()

test_df.isnull().sum()
test_df['Age'][np.isnan(test_df['Age'])] = means


titanic_df['Age'][np.isnan(titanic_df['Age'])] = titanic_mean

#convert to int

titanic_df['Age'] = titanic_df['Age'].astype(int)
test_df['Age'] = test_df['Age'].astype(int)


fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))
axis1.set_title('Original Age values - Titanic')
axis2.set_title('New Age values - Titanic')

titanic_df['Age'].hist(bins=70)

titanic_df = titanic_df.drop('Cabin', axis = 1)
test_df = test_df.drop('Cabin', axis=1)

titanic_df['Family'] = titanic_df['Parch'] + titanic_df['SibSp']

titanic_df[titanic_df['Family'] > 0]

test_df['Family'] =  test_df["Parch"] + test_df["SibSp"]

# drop Parch & SibSp
titanic_df = titanic_df.drop(['SibSp','Parch'], axis=1)
test_df    = test_df.drop(['SibSp','Parch'], axis=1)



def get_person(passenger):
    age, sex = passenger
    return 'child' if age < 16 else sex

titanic_df['Person'] = titanic_df[['Age', 'Sex']].apply(get_person, axis=1)
test_df['Person']    = test_df[['Age','Sex']].apply(get_person,axis=1)


titanic_df.drop(['Sex'],axis=1,inplace=True)
test_df.drop(['Sex'],axis=1,inplace=True)

## create dummy variables for Person column, & drop Male as it has the lowest average of survived passengers
person_dummies_titanic  = pd.get_dummies(titanic_df['Person'])
person_dummies_titanic.columns = ['Child','Female','Male']
person_dummies_titanic = person_dummies_titanic.drop('Male', axis=1)

person_dummies_test  = pd.get_dummies(test_df['Person'])
person_dummies_test.columns = ['Child','Female','Male']
person_dummies_test.drop(['Male'], axis=1, inplace=True)


titanic_df = titanic_df.join(person_dummies_titanic)
test_df    = test_df.join(person_dummies_test)


titanic_df.drop(['Person'],axis=1,inplace=True)
test_df.drop(['Person'],axis=1,inplace=True)


#PC class
# create dummy variables for Pclass column, & drop 3rd class as it has the lowest average of survived passengers
pclass_dummies_titanic  = pd.get_dummies(titanic_df['Pclass'])
pclass_dummies_titanic.columns = ['Class_1','Class_2','Class_3']
pclass_dummies_titanic.drop(['Class_3'], axis=1, inplace=True)

pclass_dummies_test  = pd.get_dummies(test_df['Pclass'])
pclass_dummies_test.columns = ['Class_1','Class_2','Class_3']
pclass_dummies_test.drop(['Class_3'], axis=1, inplace=True)

titanic_df.drop(['Pclass'],axis=1,inplace=True)
test_df.drop(['Pclass'],axis=1,inplace=True)

titanic_df = titanic_df.join(pclass_dummies_titanic)
test_df    = test_df.join(pclass_dummies_test)




# define training and testing sets

X_train = titanic_df.drop("Survived",axis=1)
Y_train = titanic_df["Survived"]
X_test  = test_df.drop("PassengerId",axis=1).copy()


# Logistic Regression

logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

#X_test['Fare'][np.isnan(X_test['Fare'])] = 35.6271884892086
Y_pred = logreg.predict(X_test)

logreg.score(X_train, Y_train)
