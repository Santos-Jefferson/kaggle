# Python ≥3.5 is required
import sys

assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn

assert sklearn.__version__ >= "0.20"
# Common imports
import numpy as np
import os

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "end_to_end_project"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


# Ignore useless warnings (see SciPy issue #5998)
import warnings

warnings.filterwarnings(action="ignore", message="^internal gelsd")

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os


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

print(test_data)
print(titanic_df)

columns_to_use = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
# titanic_df = pd.get_dummies(titanic_df[columns_to_use])

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(titanic_df, titanic_df.Sex):
    strat_train_set = titanic_df.loc[train_index]
    strat_test_set = titanic_df.loc[test_index]

strat_test_set.Sex.value_counts() / len(strat_test_set)
titanic_df.Sex.value_counts() / len(titanic_df)


def income_cat_proportions(data):
    return data["Sex"].value_counts() / len(data)


train_set, test_set = train_test_split(titanic_df, test_size=0.2, random_state=42)

compare_props = pd.DataFrame({
    "Overall": income_cat_proportions(titanic_df),
    "Stratified": income_cat_proportions(strat_test_set),
    "Random": income_cat_proportions(test_set),
}).sort_index()
compare_props["Rand. %error"] = 100 * compare_props["Random"] / compare_props["Overall"] - 100
compare_props["Strat. %error"] = 100 * compare_props["Stratified"] / compare_props["Overall"] - 100

titanic_df = strat_train_set.copy()

titanic_df['sibsp_parch'] = (titanic_df.SibSp + titanic_df.Parch)
titanic_df['fare_per_sibspparch'] = (titanic_df.Fare / (titanic_df.sibsp_parch + 1))
print(titanic_df)

titanic_df = strat_train_set.drop('Survived', axis=1)
titanic_df_labels = strat_train_set['Survived'].copy()

sample_incomplete_rows = titanic_df[titanic_df.isnull().any(axis=1)].head()
sample_incomplete_rows

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='median')
titanic_df_num = titanic_df.select_dtypes(include=[np.number])
imputer.fit(titanic_df_num)
print(imputer.statistics_)
print(titanic_df_num.median().values)

X = imputer.transform(titanic_df_num)
titanic_df_tr = pd.DataFrame(X, columns=titanic_df_num.columns, index=titanic_df.index)
titanic_df_tr.loc[sample_incomplete_rows.index.values]
imputer.strategy
titanic_df_tr = pd.DataFrame(X, columns=titanic_df_num.columns, index=titanic_df_num.index)

titanic_df_cat = titanic_df[['Sex', 'Embarked']]
titanic_df_cat.fillna(method='ffill', inplace=True)
titanic_df[['Sex', 'Embarked']] = titanic_df[['Sex', 'Embarked']].fillna(method='ffill')

from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder(sparse=False)
titanic_df_1hot = cat_encoder.fit_transform(titanic_df_cat)
titanic_df_1hot

from sklearn.base import BaseEstimator, TransformerMixin

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


# attr_adder = CombinedAttributesAdder(add_fare_per_sibspparch=False)
# titanic_df_extra_attribs = attr_adder.transform(titanic_df_num.values)
#
# titanic_df_extra_attribs = pd.DataFrame(
#     titanic_df_extra_attribs,
#     columns=list(titanic_df.columns) + ['sibsp_parch'],
#     index=titanic_df.index)
#
# titanic_df_extra_attribs.head()

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])

titanic_df_num_tr = num_pipeline.fit_transform(titanic_df_num)

from sklearn.compose import ColumnTransformer

num_attribs = list(titanic_df_num)
cat_attribs = ['Sex', 'Embarked']

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])

titanic_df_prepared = full_pipeline.fit_transform(titanic_df)
titanic_df_prepared

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC

lin_reg = SVC(degree=15, kernel='rbf', gamma=0.1 )

# kneigh = KNeighborsClassifier()
# randomForest = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=1)
# rfreg = RandomForestRegressor(n_estimators=200, random_state=42)
# rfreg.fit(titanic_df_prepared, titanic_df_labels)
# svc_class = SVC(kernel='linear')
# svc_class.fit(titanic_df_prepared, titanic_df_labels)
#
#
#
# randomForest.fit(titanic_df_prepared, titanic_df_labels)
#
# # let's try the full preprocessing pipeline on a few training instances
# some_data = titanic_df
# some_labels = titanic_df_labels
# some_data_prepared = full_pipeline.transform(some_data)
# preds = randomForest.predict(some_data_prepared)
#
# print("Predictions:", preds)
# print("Labels:", list(some_labels))
#
# from sklearn.metrics import mean_squared_error
#
# titanic_preds = svc_class.predict(titanic_df_prepared)
# svm_mse = mean_squared_error(titanic_df_labels, titanic_preds)
# svm_rmse = np.sqrt(svm_mse)
# svm_rmse
#
# titanic_preds = rfreg.predict(titanic_df_prepared)
# forest_mse = mean_squared_error(titanic_df_labels, titanic_preds)
# forest_rmse = np.sqrt(forest_mse)
# forest_rmse
#
# housing_predictions = randomForest.predict(titanic_df_prepared)
# lin_mse = mean_squared_error(titanic_df_labels, housing_predictions)
# lin_rmse = np.sqrt(lin_mse)
# lin_rmse
#
# from sklearn.metrics import mean_absolute_error
#
# lin_mae = mean_absolute_error(titanic_df_labels, housing_predictions)
# lin_mae
#
# from sklearn.tree import DecisionTreeRegressor
#
# tree_reg = DecisionTreeRegressor(random_state=42)
# tree_reg.fit(titanic_df_prepared, titanic_df_labels)
#
# titanic_predictions = tree_reg.predict(titanic_df_prepared)
# tree_mse = mean_squared_error(titanic_df_labels, titanic_predictions)
# tree_rmse = np.sqrt(tree_mse)
# tree_rmse
#
from sklearn.model_selection import cross_val_score

#
# tree_scores = cross_val_score(tree_reg, titanic_df_prepared, titanic_df_labels,
#                          scoring="neg_mean_squared_error", cv=10)
# treereg_rmse_scores = np.sqrt(-tree_scores)
#
# def display_scores(scores):
#     print('------treereg----------')
#     print("Scores:", scores)
#     print("Mean:", scores.mean())
#     print("Standard deviation:", scores.std())
#     print('----------------')
#
# display_scores(treereg_rmse_scores)
#
# ran_scores = cross_val_score(randomForest, titanic_df_prepared, titanic_df_labels,
#                          scoring="neg_mean_squared_error", cv=10)
# randomForest_rmse_scores = np.sqrt(-ran_scores)
#
# def display_scores(scores):
#     print('------randomForest----------')
#     print("Scores:", scores)
#     print("Mean:", scores.mean())
#     print("Standard deviation:", scores.std())
#     print('----------------')
#
# display_scores(randomForest_rmse_scores)
#
linreg_scores = cross_val_score(lin_reg, titanic_df_prepared, titanic_df_labels,
                                scoring="neg_mean_squared_error", cv=10)
lin_reg_rmse_scores = np.sqrt(-linreg_scores)


def display_scores(scores):
    print('------lin_reg----------')
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
    print('----------------')


display_scores(lin_reg_rmse_scores)
# sys.exit()
#
# rfreg_scores = cross_val_score(rfreg, titanic_df_prepared, titanic_df_labels,
#                          scoring="neg_mean_squared_error", cv=10)
# rfreg_rmse_scores = np.sqrt(-rfreg_scores)
#
# def display_scores(scores):
#     print('------rfreg----------')
#     print("Scores:", scores)
#     print("Mean:", scores.mean())
#     print("Standard deviation:", scores.std())
#     print('----------------')
#
# display_scores(rfreg_rmse_scores)
#
# svc_scores = cross_val_score(svc_class, titanic_df_prepared, titanic_df_labels,
#                          scoring="neg_mean_squared_error", cv=10)
# svc_rmse_scores = np.sqrt(-svc_scores)
#
# def display_scores(scores):
#     print('------svc_class----------')
#     print("Scores:", scores)
#     print("Mean:", scores.mean())
#     print("Standard deviation:", scores.std())
#     print('----------------')
#
# display_scores(svc_rmse_scores)


# from sklearn.model_selection import GridSearchCV
#
# param_grid = [
#     # try 12 (3×4) combinations of hyperparameters
#     {'n_estimators': [100, 200, 300, 400, 1000], 'max_features': [2, 4, 6, 8, 10]},
#     # then try 6 (2×3) combinations with bootstrap set as False
#     {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
# ]
#
# forest_reg = RandomForestClassifier(random_state=42)
# # train across 5 folds, that's a total of (12+6)*5=90 rounds of training
# grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
#                            scoring='neg_mean_squared_error',
#                            return_train_score=True)
# grid_search.fit(titanic_df_prepared, titanic_df_labels)
#
# print('best parameter found')
# print(grid_search.best_params_)
# print(grid_search.best_estimator_)
#
# cvres = grid_search.cv_results_
# for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
#     print(np.sqrt(-mean_score), params)
#
# pd.DataFrame(grid_search.cv_results_)

# from sklearn.model_selection import RandomizedSearchCV
# from scipy.stats import randint
#
# param_distribs = {
#     # 'C': randint(0.001, 5.0),
#     'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
#     'degree': randint(low=1, high=100),
#     # 'gamma': random.randrange(0.01, 1.0),
#     # 'shrinking ': [True, False],
#     # 'max_features': randint(low=1, high=8),
# }
#
# forest_reg = SVC()
# rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
#                                 n_iter=1, cv=5, scoring='neg_mean_squared_error', random_state=42)
# rnd_search.fit(titanic_df_prepared, titanic_df_labels)
#
# print('best parameter found')
# print(rnd_search.best_params_)
# print(rnd_search.best_estimator_)
# # sys.exit()
#
# cvres = rnd_search.cv_results_
# for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
#     print(np.sqrt(-mean_score), params)
#
# pd.DataFrame(rnd_search.cv_results_)
#
# feature_importances = rnd_search.best_estimator_.feature_importances_
# feature_importances
#
# extra_attribs = []
# # cat_encoder = cat_pipeline.named_steps["cat_encoder"] # old solution
# cat_encoder = full_pipeline.named_transformers_["cat"]
# cat_one_hot_attribs = list(cat_encoder.categories_[0])
# attributes = num_attribs + extra_attribs + cat_one_hot_attribs
# print(sorted(zip(feature_importances, attributes), reverse=True))

# final_model = grid_search.best_estimator_
# final_model = rnd_search.best_estimator_
X_test = strat_test_set.drop("Survived", axis=1)
X_test = test_data
# y_test = strat_test_set["Survived"].copy()

X_test_prepared = full_pipeline.transform(X_test)
lin_reg.fit(titanic_df_prepared, titanic_df_labels)
final_predictions = lin_reg.predict(X_test_prepared)
generate_submission_file(X_test, final_predictions, 'svc15_0.10_1.csv')

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
