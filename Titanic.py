import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

# Read the Titanic dataset
titanic_data = pd.read_csv(r"DataSets/titanic_train.csv")

# Exclude non-numeric columns from correlation analysis
numeric_columns = titanic_data.select_dtypes(include=[np.number])
correlation_matrix = numeric_columns.corr()

# Plot the correlation matrix
sns.heatmap(correlation_matrix, cmap="YlGnBu", annot=True)
plt.show()

split=StratifiedShuffleSplit(1, test_size=0.2)
for train_indices, test_indices in split.split(titanic_data, titanic_data[["Survived", "Pclass", "Sex_female", "Sex_male"]]):
  strat_train_set = titanic_data.loc[train_indices]
  strat_test_set = titanic_data.loc[test_indices]


class AgeImputer(BaseEstimator, TransformerMixin):
  def fit(self, X):
    return self
  def transform(self, X):
    imputer = SimpleImputer(strategy="mean")
    X["Age"] =  imputer.fit_transform(X[["Age"]])
    return X

class FeatureEncoder(BaseEstimator, TransformerMixin):
  def fit(self, X):
    return self
  def transform(self, X):
    ohe=OneHotEncoder()
    gendertransform=ohe.fit_transform(X[["Sex"]]).toarray()
    return X
  print(titanic_data.head())