import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

# Read the Titanic dataset
titanic_data = pd.read_csv(r"DataSets/titanic_train.csv")

# Exclude non-numeric columns from correlation analysis
numeric_columns = titanic_data.select_dtypes(include=[np.number])
correlation_matrix = numeric_columns.corr()

# Plot the correlation matrix
plt.subplot(1,3,1)
sns.heatmap(correlation_matrix, cmap="YlGnBu")
split=StratifiedShuffleSplit(1, test_size=0.2)
for train_indices, test_indices in split.split(titanic_data, titanic_data[["Survived", "Pclass", "Sex"]]):
  strat_train_set = titanic_data.loc[train_indices]
  strat_test_set = titanic_data.loc[test_indices]
plt.subplot(1,3,2)
strat_train_set["Survived"].hist()
strat_train_set["Pclass"].hist()

plt.subplot(1,3,3)
strat_test_set["Survived"].hist()
strat_test_set["Pclass"].hist()
plt.show()