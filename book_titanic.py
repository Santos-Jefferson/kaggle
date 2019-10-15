import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from pandas.plotting import scatter_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, mean_squared_error

import os

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

names = [
    # "Nearest Neighbors", "Linear SVM", "RBF SVM",
    # "Gaussian Process",
    # "Decision Tree",
    "Random Forest",
    # "Neural Net", "AdaBoost",
    # "Naive Bayes", "QDA"
]

classifiers = [
    # KNeighborsClassifier(3),
    SVC(verbose=True, random_state=42),
    # SVC(gamma=0.1, C=1),
    # GaussianProcessClassifier(1.0 * RBF(1.0)),
    # DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(verbose=1, n_jobs=-1, random_state=42),
    # MLPClassifier(alpha=1, max_iter=1000),
    # AdaBoostClassifier(),
    # GaussianNB(),
    # QuadraticDiscriminantAnalysis()
]

algos_params = {
    # "Nearest Neighbors":
    #     [
    #         {
    #             'algorithm': ['ball_tree', 'kd_tree', 'brute', 'auto'],
    #             # 'leaf_size': 30,
    #             'n_neighbors': np.arange(2, 11),
    #             'p': [1, 2],
    #             'weights': ['uniform', 'distance'],
    #         },
    #     ],
    "Linear SVM":
        [
            {
                'C': np.arange(1.0, 1.9, 0.3),
                'kernel': ['poly', 'rbf', 'sigmoid'],
                'degree': np.arange(3, 6, 1),
                'gamma': np.arange(0.001, 0.009, 0.002),
                # 'coef0': np.arange(0.0, 1.0, 0.1),
                # 'shrinking': [True, False],
                # 'decision_function_shape': ['ovo', 'ovr'],
                # 'random_state': np.arange(12, 52, 20),
                # 'verbose': True
            }
        ],
    # "Gaussian Process":
    #     [
    #         {
    #             'n_restarts_optimizer': np.arange(0, 10, 1),
    #             'max_iter_predict': np.arange(90, 150, 10),
    #             'warm_start': [True, False],
    #             'random_state': np.arange(2, 52, 10),
    #             'multi_class': ['one_vs_rest', 'one_vs_one']
    #         }
    #     ],
    # "Decision Tree":
    #     [
    #         {
    #             'criterion': ["gini", 'entropy'],
    #             'splitter': ["best", 'random'],
    #             'max_depth': np.arange(0, 10, 2),
    #             'min_samples_split': np.arange(1, 5, 1),
    #             'min_samples_leaf': np.arange(1, 5, 1),
    #             'min_weight_fraction_leaf': np.arange(0, 5, 1),
    #             'max_features': np.arange(0, 5, 1),
    #             'random_state': np.arange(2, 52, 10),
    #             'max_leaf_nodes': [np.arange(0, 5, 1), None],
    #             'min_impurity_decrease': np.arange(0, 5, 1),
    #             'presort': [True, False],
    #         }
    #     ],
    "Random Forest":
        [
            {
                'n_estimators': np.arange(100, 10000, 200),
                'criterion': ['gini', 'entropy'],
                # 'max_depth': np.arange(1, 10, 1),
                # 'min_samples_split': np.arange(2, 10, 1),
                # 'min_samples_leaf': np.arange(1, 6, 1),
                # 'min_weight_fraction_leaf': np.arange(0, 3, 1),
                'max_features': np.arange(2, 10, 2),
                # 'max_leaf_nodes': [None, np.arange(0, 2, 1)],
                # 'min_impurity_decrease': np.arange(0., 1., 0.3),
                # min_impurity_split=None,
                # 'bootstrap': [True, False],
                # 'oob_score': [True, False],
                # 'random_state': np.arange(2, 1002, 20),
                # 'verbose': [1],
                # 'warm_start': [True, False],
                # 'njobs': []
                # class_weight=None
            }
        ],
    # "Neural Net":
    #     [
    #         {
    #             'hidden_layer_sizes': [(100,), (200,), (300,)],
    #             'activation': ["relu", 'identity', 'logistic', 'tanh'],
    #             'solver': ['adam', 'lbfgs', 'sgd'],
    #             'alpha': np.arange(0.0001, 0.1, 0.01),
    #             # batch_size='auto',
    #             'learning_rate': ["constant", 'invscaling', 'adaptive'],
    #             'learning_rate_init': np.arange(0.001, 0.1, 0.01),
    #             # power_t=0.5,
    #             'max_iter': np.arange(200, 500, 50),
    #             # shuffle=True,
    #             'random_state': np.arange(2, 52, 10),
    #             # 'tol': 1e-4,
    #             # verbose=False, warm_start=False,
    #             'momentum': np.arange(0.1, 1.0, 0.1),
    #             # nesterovs_momentum=True, early_stopping=False,
    #             # validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
    #             # epsilon=1e-8, n_iter_no_change=10):
    #         }
    #     ],
    # "AdaBoost":
    #     [
    #         {
    #             # 'base_estimator':=None,
    #             'n_estimators': np.arange(50, 500, 10),
    #             'learning_rate': np.arange(1.0, 3.0, 0.2),
    #             'algorithm': ['SAMME.R', 'SAMME'],
    #             'random_state': np.arange(2, 52, 10),
    #         }
    #
    #     ],
    # "Naive Bayes":
    #     [
    #         {
    #
    #         }
    #     ],
    # "QDA":
    #     [
    #         {
    #             # priors=None,
    #             'reg_param': np.arange(0., 1., 0.2),
    #             'store_covariance': [False, True],
    #             # tol=1.0e-4, store_covariances=None
    #         }
    #     ],
}


def generate_submission_file(testdata, predictions, filename):
    output = pd.DataFrame({'PassengerId': testdata.PassengerId, 'Survived': predictions})
    output.to_csv(filename, index=False)
    print("Your submission was successfully saved!")


for dirname, _, filenames in os.walk('input/titanic/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train_data = pd.read_csv('input/titanic/train.csv')
test_data = pd.read_csv('input/titanic/test.csv')

print(train_data.head())
print(train_data.info())
print(train_data.describe().to_string())

train_data["AgeBucket"] = train_data["Age"] // 15 * 15
test_data["AgeBucket"] = test_data["Age"] // 15 * 15
print(train_data[["AgeBucket", "Survived"]].groupby(['AgeBucket']).mean())

train_data["RelativesOnboard"] = train_data["SibSp"] + test_data["Parch"]
test_data["RelativesOnboard"] = test_data["SibSp"] + test_data["Parch"]
print(train_data[["RelativesOnboard", "Survived"]].groupby(['RelativesOnboard']).mean())

from sklearn.base import BaseEstimator, TransformerMixin


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names]


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

num_pipeline = Pipeline([
    ("select_numeric", DataFrameSelector(["AgeBucket", "RelativesOnboard", "Fare"])),
    ("imputer", SimpleImputer(strategy="median")),
])

print(num_pipeline.fit_transform(train_data))


# Inspired from stackoverflow.com/questions/25239958
class MostFrequentImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X],
                                        index=X.columns)
        print(self.most_frequent_.to_string())
        return self

    def transform(self, X, y=None):
        # print(X.fillna(self.most_frequent_))
        return X.fillna(self.most_frequent_)


from sklearn.preprocessing import OneHotEncoder

cat_pipeline = Pipeline([
    ("select_cat", DataFrameSelector(["Pclass", "Sex", "Embarked"])),
    ("imputer", MostFrequentImputer()),
    ("cat_encoder", OneHotEncoder(sparse=False)),
])

print(cat_pipeline.fit_transform(train_data))

from sklearn.pipeline import FeatureUnion

preprocess_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline),
])

X_train = preprocess_pipeline.fit_transform(train_data)
print(X_train)

y_train = train_data["Survived"]
print(y_train)

toplot = []
best_estimators = []

for clf, name, (key, value) in zip(classifiers, names, algos_params.items()):
    clf.fit(X_train, y_train)

    grid_search = GridSearchCV(clf, value, cv=5,/
                               scoring='neg_mean_squared_error',
                               return_train_score=True)
    grid_search.fit(X_train, y_train)

    print('--------------------------best parameter found-----------------------------')
    print(grid_search.best_params_)
    print(grid_search.best_estimator_)

    cvres = grid_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)

    grid_results = pd.DataFrame(grid_search.cv_results_)
    grid_results.sort_values('rank_test_score', inplace=True)
    grid_results.to_csv(key + '_results.csv')
    best_estimators.append(name)
    best_estimators.append(grid_search.best_params_)
    best_estimators.append(grid_search.best_estimator_)
    print(grid_results)
print(best_estimators)
# final_model = grid_search.best_estimator_
# X_test = preprocess_pipeline.transform(test_data)
# final_predictions = final_model.predict(X_test)
# generate_submission_file(test_data, final_predictions, 'finalSub.csv')

#     y_pred = clf.predict(X_test)
#     clf_scores = cross_val_score(clf, X_train, y_train, cv=10)
#     print('---' + name + '---')
#     print(clf_scores.mean())
#     print()
#     toplot.append(clf_scores)
#     generate_submission_file(test_data, y_pred, name + '_' + str(clf_scores.mean()) + '_.csv')
#     plt.figure(figsize=(8, 4))
#     plt.plot([1] * 10, clf_scores, ".")
# # plt.plot([2]*10, clf_scores, ".")
# plt.boxplot(toplot, labels=names)
# plt.ylabel("Accuracy", fontsize=14)
# plt.show()
