import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix
from sklearn.naive_bayes import GaussianNB

import os

for dirname, _, filenames in os.walk('input/titanic/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

subm_file = pd.read_csv('input/titanic/gender_submission.csv')
test_data = pd.read_csv('input/titanic/test.csv')
train_data = pd.read_csv('input/titanic/train.csv',
                         usecols=['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare'])
train_data.dropna(inplace=True)
train_data.Sex = np.where(train_data.Sex.str.contains('female'), 0, 1)
print(train_data.head().to_string())

X = train_data.drop('Survived', axis=1)
y = train_data[['Survived']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

attributes = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
scatter_matrix(train_data[attributes], figsize=(12, 8))
# plt.show()


clf = GaussianNB()
clf.fit(X_train, y_train)
y_pred = clf.predict_proba(X_test)

accuracy_score = accuracy_score(y_test, y_pred.round(), normalize=False)
print(accuracy_score * 100)

