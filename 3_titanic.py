# IMPORTS #########################################################
import sys
import numpy as np
import os

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt

# Ignore useless warnings (see SciPy issue #5998)
import warnings

warnings.filterwarnings(action="ignore", message="^internal gelsd")

import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.base import BaseEstimator, TransformerMixin

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# CONSTANTS ######################################################
# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "end_to_end_project"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

# CLASSES #########################################################
# COMBINE ATTRIBUTES #
# column index
sibsp_ix, parch_ix, fare_ix = 3, 4, 5


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_fare_per_sibspparch=True):  # no *args or **kargs
        self.add_fare_per_sibspparch = add_fare_per_sibspparch

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X, y=None):
        sibsp_parch = X[:, sibsp_ix] + X[:, parch_ix]
        if self.add_fare_per_sibspparch:
            fare_per_sibspparch = X[:, fare_ix] / (X[:, sibsp_ix] + X[:, parch_ix] + 1)
            return np.c_[X, sibsp_parch, fare_per_sibspparch]
        else:
            return np.c_[X, sibsp_parch]


# FUNCTIONS #######################################################
def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


def generate_submission_file(testdata, predictions, filename):
    # predictions = np.where(predictions > 0.5, 1, 0)
    output = pd.DataFrame({'PassengerId': testdata.PassengerId, 'Survived': predictions})
    output.to_csv(filename, index=False)
    print("Your submission was successfully saved!")


for dirname, _, filenames in os.walk('input/titanic/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

test_data = pd.read_csv('input/titanic/test.csv')
titanic_df = pd.read_csv('input/titanic/train.csv')

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(titanic_df, titanic_df.Sex):
    strat_train_set = titanic_df.loc[train_index]
    strat_test_set = titanic_df.loc[test_index]

titanic_df = strat_train_set.copy()

titanic_df['sibsp_parch'] = (titanic_df.SibSp + titanic_df.Parch)
titanic_df['fare_per_sibspparch'] = (titanic_df.Fare / (titanic_df.sibsp_parch + 1))
# print(titanic_df)

titanic_df = strat_train_set.drop('Survived', axis=1)
titanic_df_labels = strat_train_set['Survived'].copy()

sample_incomplete_rows = titanic_df[titanic_df.isnull().any(axis=1)].head()
print('sample incomplete rows')
print(sample_incomplete_rows)

# SIMPLEIMPUTER ########################################################
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='median')
titanic_df_num = titanic_df.select_dtypes(include=[np.number])  # Get number columns from dataset
imputer.fit(titanic_df_num)
print(imputer.statistics_)
print(titanic_df_num.median().values)

X = imputer.transform(titanic_df_num)
titanic_df_tr = pd.DataFrame(X, columns=titanic_df_num.columns, index=titanic_df.index)
# titanic_df_tr.loc[sample_incomplete_rows.index.values]
print('imputer strategy')
print(imputer.strategy)
titanic_df_tr = pd.DataFrame(X, columns=titanic_df_num.columns, index=titanic_df_num.index)

categories = ['Sex', 'Embarked']

titanic_df_cat = titanic_df[categories]
titanic_df_cat.fillna(method='ffill', inplace=True)
titanic_df[categories] = titanic_df[categories].fillna(method='ffill')

# ONEHOTENCODER ##########################################################
from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder(sparse=False)
titanic_df_1hot = cat_encoder.fit_transform(titanic_df_cat)
print('one hot encoder with category columns')
print(titanic_df_1hot)


# PIPELINE ############################################################
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])

titanic_df_num_tr = num_pipeline.fit_transform(titanic_df_num)

# COLUMNTRANSFORMER (NUMBERS / CATEGORIES) ##############################

from sklearn.compose import ColumnTransformer

num_attribs = list(titanic_df_num)
cat_attribs = categories

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])

titanic_df_prepared = full_pipeline.fit_transform(titanic_df)
print('df prepared')
print(titanic_df_prepared)

# ML ALGOS ##########################################################

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from scipy.stats import randint

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(algorithm='ball_tree', n_neighbors=19, p=1, weights='distance'),
    SVC(C=1.6, coef0=0.30000000000000004, decision_function_shape='ovr', degree=2, gamma=0.06099999999999999, kernel='rbf', random_state=12, shrinking=True),
    # # SVC(gamma=0.1, C=1),
    # GaussianProcessClassifier(1.0 * RBF(1.0)),
    # DecisionTreeClassifier(max_depth=5),
    # RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    # MLPClassifier(alpha=1, max_iter=1000),
    # AdaBoostClassifier(),
    # GaussianNB(),
    # QuadraticDiscriminantAnalysis()
]




for clf, name in zip(classifiers, names):
    clf.fit(titanic_df_prepared, titanic_df_labels)

    titanic_predictions = clf.predict(titanic_df_prepared)
    mse = mean_squared_error(titanic_df_labels, titanic_predictions)
    rmse = np.sqrt(mse)
    print()
    print()
    print('---_' + name + '_RMSE_------------------------------------')
    print(rmse)
    print('***********************************************************')

    mae = mean_absolute_error(titanic_df_labels, titanic_predictions)
    print('---_' + name + '_MAE_------------------------------------')
    print(mae)
    print('***********************************************************')


    clf_scores = cross_val_score(clf, titanic_df_prepared, titanic_df_labels,
                                 scoring="neg_mean_squared_error", cv=10)
    clf_rmse_scores = np.sqrt(-clf_scores)


    def display_scores(scores):
        print('------_' + name + '_-------')
        print("Scores:", scores)
        print("Mean:", scores.mean())
        print("Standard deviation:", scores.std())
        print('=============================================================')
        print('=============================================================')
        print('=============================================================')
        print()
        print()


    display_scores(clf_rmse_scores)



    X_test = strat_test_set.drop("Survived", axis=1)
    X_test = test_data
    # y_test = strat_test_set["Survived"].copy()

    X_test_prepared = full_pipeline.transform(X_test)
    clf.fit(titanic_df_prepared, titanic_df_labels)
    final_predictions = clf.predict(X_test_prepared)
    generate_submission_file(X_test, final_predictions, name + '_subm.csv')

# final_mse = mean_squared_error(y_test, final_predictions)
# final_rmse = np.sqrt(final_mse)


# from scipy import stats
#
# confidence = 0.95
# squared_errors = (final_predictions - y_test) ** 2
# np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
#                          loc=squared_errors.mean(),
#                          scale=stats.sem(squared_errors)))
#
# m = len(squared_errors)
# mean = squared_errors.mean()
# tscore = stats.t.ppf((1 + confidence) / 2, df=m - 1)
# tmargin = tscore * squared_errors.std(ddof=1) / np.sqrt(m)
# print(np.sqrt(mean - tmargin), np.sqrt(mean + tmargin))
#
# zscore = stats.norm.ppf((1 + confidence) / 2)
# zmargin = zscore * squared_errors.std(ddof=1) / np.sqrt(m)
# print(np.sqrt(mean - zmargin), np.sqrt(mean + zmargin))
#
#
