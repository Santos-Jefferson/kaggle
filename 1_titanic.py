import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from pandas.plotting import scatter_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import os


def generate_submission_file(testdata, predictions, filename):
    output = pd.DataFrame({'PassengerId': testdata.PassengerId, 'Survived': predictions})
    output.to_csv(filename, index=False)
    print("Your submission was successfully saved!")


for dirname, _, filenames in os.walk('input/titanic/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

test_data = pd.read_csv('input/titanic/test.csv')
test_data_orig = test_data.copy()
# test_data.Sex = np.where(test_data.Sex.str.contains('female'), 0, 1)

train_data = pd.read_csv('input/titanic/train.csv')
# train_data.dropna(inplace=True)
# train_data.Sex = np.where(train_data.Sex.str.contains('female'), 0, 1)

features = [
    'Pclass',
    'Sex',
    'Age',
    'SibSp',
    'Parch',
    'Fare'
]

# X = train_data.drop('Survived', axis=1)
X = pd.get_dummies(train_data[features])
X.fillna(X.mean(), inplace=True)
X_test2 = pd.get_dummies(test_data[features])
X_test2.fillna(X_test2.mean(), inplace=True)
y = train_data[['Survived']]

print(X.corr())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

print(X.info())
print(X_test2.info())

print(y.info())
print(X_train.info())
print(y_train.info())

acc = []
cv_scores = []
neighbors = list(range(1, 50, 2))
for i in neighbors:
    clf = KNeighborsClassifier(n_neighbors=i)
    scores = cross_val_score(clf, X_train, y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())

mse = [1 - x for x in cv_scores]
# determining best k
optimal_k = neighbors[mse.index(min(mse))]
print(optimal_k)

clf = KNeighborsClassifier(n_neighbors=optimal_k)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test2)

generate_submission_file(test_data_orig, y_pred, 'knn5.csv')
