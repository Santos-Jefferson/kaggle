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

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

lin_reg = LinearRegression()
lin_reg = KNeighborsClassifier()
lin_reg = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=1)
lin_reg.fit(titanic_df_prepared, titanic_df_labels)

# let's try the full preprocessing pipeline on a few training instances
some_data = titanic_df
some_labels = titanic_df_labels
some_data_prepared = full_pipeline.transform(some_data)
preds = lin_reg.predict(some_data_prepared)

print("Predictions:", preds)
print("Labels:", list(some_labels))

from sklearn.metrics import mean_squared_error

housing_predictions = lin_reg.predict(titanic_df_prepared)
lin_mse = mean_squared_error(titanic_df_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse

from sklearn.metrics import mean_absolute_error

lin_mae = mean_absolute_error(titanic_df_labels, housing_predictions)
lin_mae

from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(titanic_df_prepared, titanic_df_labels)

titanic_predictions = tree_reg.predict(titanic_df_prepared)
tree_mse = mean_squared_error(titanic_df_labels, titanic_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse


from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, titanic_df_prepared, titanic_df_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)
def display_scores(scores):
    print()
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
    print()

display_scores(tree_rmse_scores)

lin_scores = cross_val_score(lin_reg, titanic_df_prepared, titanic_df_labels,
                             scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)

from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
forest_reg.fit(titanic_df_prepared, titanic_df_labels)

titanic_predictions = forest_reg.predict(titanic_df_prepared)
forest_mse = mean_squared_error(titanic_df_labels, titanic_predictions)
forest_rmse = np.sqrt(forest_mse)
print(forest_rmse)

from sklearn.model_selection import cross_val_score

forest_scores = cross_val_score(forest_reg, titanic_df_prepared, titanic_df_labels,
                                scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)

scores = cross_val_score(lin_reg, titanic_df_prepared, titanic_df_labels, scoring="neg_mean_squared_error", cv=10)
pd.Series(np.sqrt(-scores)).describe()


from sklearn.svm import SVR

svm_reg = SVR(kernel="linear")
svm_reg.fit(titanic_df_prepared, titanic_df_labels)
titanic_predictions = svm_reg.predict(titanic_df_prepared)
svm_mse = mean_squared_error(titanic_df_labels, titanic_predictions)
svm_rmse = np.sqrt(svm_mse)
print(svm_rmse)