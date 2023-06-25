import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv("/kaggle/input/mall-customerscsv/Mall_Customers.csv")
data.head()
data.describe()

data.columns = ["id","gender","age","annual_income","spending_score"]
data.drop(["id"],axis=1,inplace=True)
data.gender = [1 if each == "Male"  else 0 for each in data.gender]

#data normalization
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(data)
scaled = scaler.fit_transform(data)
scaled_df = pd.DataFrame(scaled, columns=data.columns)
print(scaled_df)

y = scaled_df.gender.values
x = scaled_df.drop(["gender"], axis=1) 
print(x,y)

#training model
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.35, random_state=42)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train,y_train)

print("test accuracy {}   ".format(lr.score(x_test,y_test)))
print("train accuracy {}   ".format(lr.score(x_train,y_train)))
