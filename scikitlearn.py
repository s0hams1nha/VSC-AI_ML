from sklearn.datasets import load_boston
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import pandas as pd


pipe= Pipeline([
  ("scale", StandardScaler()),
  ("model", KNeighborsRegressor())
])


mod = GridSearchCV(
  estimator=pipe,
  param_grid={'model__n_neighbors': [1,2,3,4,5,6,7,8,9,10]},
  cv=3
)


X, y= load_boston(return_X_y=True)
mod.fit(X, y)
print(pd.DataFrame(mod.cv_results_))