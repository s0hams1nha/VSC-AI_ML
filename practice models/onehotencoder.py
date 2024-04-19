import pandas as pd
from sklearn.preprocessing import OneHotEncoder

d = {'sales': [100000,222000,1000000,522000,111111,222222,1111111,20000,75000,90000,1000000,10000],

      'city': ['Tampa','Tampa','Orlando','Jacksonville','Miami','Jacksonville','Miami','Miami','Orlando','Orlando','Orlando','Orlando'],

      'size': ['Small', 'Medium','Large','Large','Small','Medium','Large','Small','Medium','Medium','Medium','Small',]}

df=pd.DataFrame(d)

ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).set_output(transform="pandas")
citytransform = ohe.fit_transform(df[["city"]])
df= pd.concat([df, citytransform], axis=1).drop(columns=["city"])

sizetransform= ohe.fit_transform(df[["size"]])
df = pd.concat([df, sizetransform], axis=1)
print(df)