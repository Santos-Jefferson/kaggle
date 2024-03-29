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

# strat_test_set.Sex.value_counts() / len(strat_test_set)
# titanic_df.Sex.value_counts() / len(titanic_df)


# def income_cat_proportions(data):
#     return data["Sex"].value_counts() / len(data)

# train_set, test_set = train_test_split(titanic_df, test_size=0.2, random_state=42)

# compare_props = pd.DataFrame({
#     "Overall": income_cat_proportions(titanic_df),
#     "Stratified": income_cat_proportions(strat_test_set),
#     "Random": income_cat_proportions(test_set),
# }).sort_index()
# compare_props["Rand. %error"] = 100 * compare_props["Random"] / compare_props["Overall"] - 100
# compare_props["Strat. %error"] = 100 * compare_props["Stratified"] / compare_props["Overall"] - 100

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

# attr_adder = CombinedAttributesAdder(add_fare_per_sibspparch=False)
# titanic_df_extra_attribs = attr_adder.transform(titanic_df_num.values)
#
# titanic_df_extra_attribs = pd.DataFrame(
#     titanic_df_extra_attribs,
#     columns=list(titanic_df.columns) + ['sibsp_parch'],
#     index=titanic_df.index)
#
# titanic_df_extra_attribs.head()

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
    KNeighborsClassifier(3),
    SVC(verbose=True),
    # SVC(gamma=0.1, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()
]

algos_params = {
    "Nearest Neighbors":
        [
            {
                'algorithm': ['ball_tree', 'kd_tree', 'brute', 'auto'],
                # 'leaf_size': 30,
                'n_neighbors': np.arange(15, 20),
                'p': [1, 2],
                'weights': ['uniform', 'distance'],
            },
        ],
    "Linear SVM":
        [
            {
                'C': np.arange(1.0, 1.9, 0.3),
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'degree': np.arange(2, 6, 1),
                'gamma': np.arange(0.001, 0.1, 0.01),
                'coef0': np.arange(0.0, 1.0, 0.1),
                'shrinking': [True, False],
                'decision_function_shape': ['ovo', 'ovr'],
                'random_state': np.arange(12, 52, 20),
                # 'verbose': True
            }
        ],
    "Gaussian Process":
        [
            {
                'n_restarts_optimizer': np.arange(0, 10, 1),
                'max_iter_predict': np.arange(90, 150, 10),
                'warm_start': [True, False],
                'random_state': np.arange(2, 52, 10),
                'multi_class': ['one_vs_rest', 'one_vs_one']
            }
        ],
    "Decision Tree":
        [
            {
                'criterion': ["gini", 'entropy'],
                'splitter': ["best", 'random'],
                'max_depth': np.arange(0, 10, 2),
                'min_samples_split': np.arange(1, 5, 1),
                'min_samples_leaf': np.arange(1, 5, 1),
                'min_weight_fraction_leaf': np.arange(0, 5, 1),
                'max_features': np.arange(0, 5, 1),
                'random_state': np.arange(2, 52, 10),
                'max_leaf_nodes': [np.arange(0, 5, 1), None],
                'min_impurity_decrease': np.arange(0, 5, 1),
                'presort': [True, False],
            }
        ],
    "Random Forest":
        [
            {
                'n_estimators': np.arange(100, 1000, 100),
                'criterion': ["gini", 'entropy'],
                'max_depth': np.arange(0, 10, 2),
                'min_samples_split': np.arange(1, 5, 1),
                'min_samples_leaf': np.arange(1, 5, 1),
                'min_weight_fraction_leaf': np.arange(0, 5, 1),
                'max_features': [2, 4, 6, 8],
                'max_leaf_nodes': np.arange(0, 5, 1),
                'min_impurity_decrease': np.arange(0, 5, 1),
                # min_impurity_split=None,
                'bootstrap': [True, False],
                'oob_score': [True, False],
                'random_state': np.arange(2, 52, 10),
                # verbose=0,
                'warm_start': [True, False],
                # class_weight=None
            }
        ],
    "Neural Net":
        [
            {
                'hidden_layer_sizes': [(100,), (200,), (300,)],
                'activation': ["relu", 'identity', 'logistic', 'tanh'],
                'solver': ['adam', 'lbfgs', 'sgd'],
                'alpha': np.arange(0.0001, 0.1, 0.01),
                # batch_size='auto',
                'learning_rate': ["constant", 'invscaling', 'adaptive'],
                'learning_rate_init': np.arange(0.001, 0.1, 0.01),
                # power_t=0.5,
                'max_iter': np.arange(200, 500, 50),
                # shuffle=True,
                'random_state': np.arange(2, 52, 10),
                # 'tol': 1e-4,
                # verbose=False, warm_start=False,
                'momentum': np.arange(0.1, 1.0, 0.1),
                # nesterovs_momentum=True, early_stopping=False,
                # validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
                # epsilon=1e-8, n_iter_no_change=10):
            }
        ],
    "AdaBoost":
        [
            {
                # 'base_estimator':=None,
                'n_estimators': np.arange(50, 500, 10),
                'learning_rate': np.arange(1.0, 3.0, 0.2),
                'algorithm': ['SAMME.R', 'SAMME'],
                'random_state': np.arange(2, 52, 10),
            }

        ],
    "Naive Bayes":
        [
            {

            }
        ],
    "QDA":
        [
            {
                # priors=None,
                'reg_param': np.arange(0., 1., 0.2),
                'store_covariance': [False, True],
                # tol=1.0e-4, store_covariances=None
            }
        ],
}

for clf, name, (key, value) in zip(classifiers, names, algos_params.items()):
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

    # # let's try the full preprocessing pipeline on a few training instances
    # some_data = titanic_df[:5]
    # some_labels = titanic_df_labels[:5]
    # some_data_prepared = full_pipeline.transform(some_data)
    # preds = clf.predict(some_data_prepared)
    #
    # print("Predictions_" + name + '_:', preds)
    # print("Labels_" + name + '_:', list(some_labels))

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

    # param_grid = [
    #     # try 12 (3×4) combinations of hyperparameters
    #     {'n_estimators': [100, 200, 300, 400, 1000], 'max_features': [2, 4, 6, 8, 10]},
    #     # then try 6 (2×3) combinations with bootstrap set as False
    #     {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
    # ]

    clf_grid = clf

    print(key)
    print(value)
    # train across 5 folds, that's a total of (12+6)*5=90 rounds of training
    grid_search = GridSearchCV(clf_grid, value, cv=5,
                               scoring='neg_mean_squared_error',
                               return_train_score=True)
    grid_search.fit(titanic_df_prepared, titanic_df_labels)

    print('--------------------------best parameter found-----------------------------')
    print(grid_search.best_params_)
    print(grid_search.best_estimator_)

    cvres = grid_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)

    grid_results = pd.DataFrame(grid_search.cv_results_)
    grid_results.sort_values('rank_test_score', inplace=True)
    grid_results.to_csv(key + '_results.csv')
    print(grid_results)
sys.exit()

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
