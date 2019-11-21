import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, cross_val_predict
from pandas.plotting import scatter_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, mean_squared_error

import os

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

names = [
    # "Nearest Neighbors",
    "Linear SVM",
    # "RBF SVM",
    # "Gaussian Process",
    # "Decision Tree",
    # "Random Forest",
    # "Neural Net", "AdaBoost",
    # "Naive Bayes", "QDA"
]

classifiers = [
    # KNeighborsClassifier(3),
    SVC(verbose=False),
    # SVC(gamma=0.1, C=1),
    # GaussianProcessClassifier(1.0 * RBF(1.0)),
    # DecisionTreeClassifier(max_depth=5),
    # RandomForestClassifier(random_state=42, verbose=1, n_jobs=-1),
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
    # "Linear SVM":
    #     [
    #         {
    #             'C': np.arange(1.0, 1.9, 0.3),
    #             'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    #             'degree': np.arange(2, 6, 1),
    #             'gamma': np.arange(0.001, 0.1, 0.01),
    #             'coef0': np.arange(0.0, 1.0, 0.1),
    #             'shrinking': [True, False],
    #             'decision_function_shape': ['ovo', 'ovr'],
    #             'random_state': np.arange(12, 52, 20),
    #             # 'verbose': True
    #         }
    #     ],
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
                'n_estimators': np.arange(100, 1000, 150),
                'criterion': ['gini', 'entropy'],
                'max_depth': np.arange(1, 10, 1),
                'min_samples_split': np.arange(2, 10, 1),
                'min_samples_leaf': np.arange(1, 6, 1),
                # 'min_weight_fraction_leaf': np.arange(0, 3, 1),
                'max_features': np.arange(2, 20, 3),
                # 'max_leaf_nodes': [None, np.arange(0, 2, 1)],
                'min_impurity_decrease': np.arange(0., 1., 0.3),
                # min_impurity_split=None,
                # 'bootstrap': [True, False],
                # 'oob_score': [True, False],
                'random_state': np.arange(12, 52, 10),
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

from sklearn.ensemble import RandomForestClassifier

clf = SVC(C=28.199999999999996, gamma=0.004)
clf.fit(X_train, y_train)
forest_scores = cross_val_score(clf, X_train, y_train, cv=10)
y_train_pred = cross_val_predict(clf, X_train, y_train, cv=3)
print(forest_scores.mean())
X_test = preprocess_pipeline.transform(test_data)
#
y_pred = clf.predict(X_test)

from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_train, y_train_pred)
print(conf_matrix)

from sklearn.metrics import precision_score, recall_score

precision = precision_score(y_train, y_train_pred)
recall = recall_score(y_train, y_train_pred)
print('precision')
print(precision)
print('recall')
print(recall)

from sklearn.metrics import f1_score

f1_score = f1_score(y_train, y_train_pred)
print('f1')
print(f1_score)

y_scores = cross_val_predict(clf, X_train, y_train, cv=3,
                             method="decision_function")

from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores)

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.legend(loc="center right", fontsize=16) # Not shown in the book
    plt.xlabel("Threshold", fontsize=16)        # Not shown
    plt.grid(True)                              # Not shown
    plt.axis([-50000, 50000, 0, 1])             # Not shown

plt.figure(figsize=(8, 4))                      # Not shown
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.plot([7813, 7813], [0., 0.9], "r:")         # Not shown
plt.plot([-50000, 7813], [0.9, 0.9], "r:")      # Not shown
plt.plot([-50000, 7813], [0.4368, 0.4368], "r:")# Not shown
plt.plot([7813], [0.9], "ro")                   # Not shown
plt.plot([7813], [0.4368], "ro")                # Not shown
# save_fig("precision_recall_vs_threshold_plot")  # Not shown
plt.show()

def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])
    plt.grid(True)

plt.figure(figsize=(8, 6))
plot_precision_vs_recall(precisions, recalls)
plt.plot([0.4368, 0.4368], [0., 0.9], "r:")
plt.plot([0.0, 0.4368], [0.9, 0.9], "r:")
plt.plot([0.4368], [0.9], "ro")
# save_fig("precision_vs_recall_plot")
plt.show()

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_train, y_scores)

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal
    plt.axis([0, 1, 0, 1])                                    # Not shown in the book
    plt.xlabel('False Positive Rate (Fall-Out)', fontsize=16) # Not shown
    plt.ylabel('True Positive Rate (Recall)', fontsize=16)    # Not shown
    plt.grid(True)                                            # Not shown

plt.figure(figsize=(8, 6))                         # Not shown
plot_roc_curve(fpr, tpr)
plt.plot([4.837e-3, 4.837e-3], [0., 0.4368], "r:") # Not shown
plt.plot([0.0, 4.837e-3], [0.4368, 0.4368], "r:")  # Not shown
plt.plot([4.837e-3], [0.4368], "ro")               # Not shown
# save_fig("roc_curve_plot")                         # Not shown
plt.show()

from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train, cv=3,
                                    method="predict_proba")

y_scores_forest = y_probas_forest[:, 1] # score = proba of positive class
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train,y_scores_forest)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, "b:", linewidth=2, label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.plot([4.837e-3, 4.837e-3], [0., 0.4368], "r:")
plt.plot([0.0, 4.837e-3], [0.4368, 0.4368], "r:")
plt.plot([4.837e-3], [0.4368], "ro")
plt.plot([4.837e-3, 4.837e-3], [0., 0.9487], "r:")
plt.plot([4.837e-3], [0.9487], "ro")
plt.grid(True)
plt.legend(loc="lower right", fontsize=16)
# save_fig("roc_curve_comparison_plot")
plt.show()

# clf_scores = cross_val_score(clf, X_train, y_train, cv=10)
# print('---rf---')
# print(clf_scores.mean())
# print()
# toplot.append(clf_scores)
generate_submission_file(test_data, y_pred, 'rf_test5.csv')
# plt.figure(figsize=(8, 4))
# plt.plot([1] * 10, clf_scores, ".")
# # plt.plot([2]*10, clf_scores, ".")
# plt.boxplot(toplot, labels=names)
# plt.ylabel("Accuracy", fontsize=14)
# plt.show()
