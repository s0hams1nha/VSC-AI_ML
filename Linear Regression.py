#Using Head Size/ Brain Weigth dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('DataSets\headbrain.csv')
data.head()
x=data['Head Size(cm^3)'].values
y=data['Brain Weight(grams)'].values
np.corrcoef(x, y)
plt.scatter(x, y, c='green', label='Data points')
plt.xlabel('Head Size (cm^3)')
plt.ylabel('Brain Weight(grams)')
plt.legend()
plt.show()
meanx=np.mean(x)
meany=np.mean(y)
n=len(x)
numer=0
denom=0
for i in range(n):
  numer+=(x[i] - meanx)*(y[i] - meany)
  denom+=(x[i] - meanx)**2
b=numer/denom
a=meany - (b*meanx)
print("Coeffs of Reggression", a, b)
plt.rcParams['figure.figsize'] = (10,5)
Y = a + (b*x)
plt.plot(x, Y, color='red', label='Regression Line')
plt.scatter(x, y, color='green', label='Scatter data')
plt.xlabel('Head Size in cm3')
plt.ylabel('Brain Weight in cm3')
plt.legend()
plt.show()
rmse=0
for i in range(n):
  rmse += (Y[i] - y[i])**2
rmse=np.sqrt(rmse/n)
print("Root Mean Square Error is", rmse)