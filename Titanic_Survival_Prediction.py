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


#impute missin values

#titanic_data["Age"]=titanic_data["Age"].fillna(titanic_data["Age"].mean())
imputer=SimpleImputer(strategy="mean")
titanic_data["Age"]=imputer.fit_transform(titanic_data[["Age"]])

#encode gender values 
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).set_output(transform="pandas")
gendertransform=ohe.fit_transform(titanic_data[["Sex"]])
titanic_data=pd.concat([titanic_data, gendertransform], axis=1).drop(columns=["Sex"])

#encode embarked column
#embarkedtransform=ohe.fit_transform(titanic_data[["Embarked"]])
#titanic_data=pd.concat([titanic_data, embarkedtransform], axis=1).drop(columns=["Embarked"])


#drop few features of the data set
titanic_data=titanic_data.drop(titanic_data[["Name", "Ticket", "Cabin", "Embarked"]], axis=1)

# Exclude non-numeric columns from correlation analysis
numeric_columns = titanic_data.select_dtypes(include=[np.number])
correlation_matrix = numeric_columns.corr()

# Plot the correlation matrix
sns.heatmap(correlation_matrix, cmap="YlGnBu", annot=True)
plt.show()


print(titanic_data.head())
print(titanic_data.info())


split=StratifiedShuffleSplit(1, test_size=0.2)
for train_indices, test_indices in split.split(titanic_data, titanic_data[["Survived", "Pclass", "Sex_female", "Sex_male"]]):
  strat_train_set = titanic_data.loc[train_indices]
  strat_test_set = titanic_data.loc[test_indices]
plt.subplot(1,3,2)
strat_train_set["Survived"].hist()
strat_train_set["Pclass"].hist()
plt.subplot(1,3,3)
strat_test_set["Survived"].hist()
strat_test_set["Pclass"].hist()

pipeline = Pipeline([()])