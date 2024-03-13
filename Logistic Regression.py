import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


data=pd.read_csv("DataSets\insurance_data.csv")

plt.scatter(data.age, data.bought_insurance, color="red")

X_train, X_test, y_train, y_test = train_test_split(data[["age"]], data.bought_insurance, train_size=0.75)

model=LogisticRegression()
model.fit(X_train, y_train)
print(model.predict(X_test))
print(model.score(X_test, y_test))