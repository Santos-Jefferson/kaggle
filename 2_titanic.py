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

strat_test_set.Sex.value_counts()/len(strat_test_set)
titanic_df.Sex.value_counts()/len(titanic_df)


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

# titanic_df.plot(kind="scatter", x="Sex_female", y="Fare", alpha=0.4,
#     s=titanic_df["Age"].sum()/100, label="Age", figsize=(10,7),
#     c="Survived", cmap=plt.get_cmap("jet"), colorbar=True,
#     sharex=False)
# plt.legend()

# from pandas.plotting import scatter_matrix
# attributes = ['Survived','Pclass', 'Sex_female', 'Age', 'SibSp', 'Fare', 'Embarked_C']
# scatter_matrix(titanic_df[attributes], figsize=(12, 8), )

titanic_df = strat_train_set.drop('Survived', axis=1)
titanic_df_labels = strat_train_set['Survived'].copy()

sample_incomplete_rows = titanic_df[titanic_df.isnull().any(axis=1)].head()
sample_incomplete_rows

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')











