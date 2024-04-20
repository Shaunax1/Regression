import pandas as pd 
import matplotlib.pyplot as plt

data = pd.read_csv("/kaggle/input/iris/Iris.csv")
data.columns =["id","sepal_length","sepal_width","petal_length","petal_width","species"]
data.drop(["id"],axis=1)
data.isnull().sum()

#plot
plt.scatter(data.sepal_length,data.petal_length)
plt.xlabel("x")
plt.ylabel("y")
plt.show()

#%%
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

x = data.iloc[:,0].values.reshape(-1,1)
y = data.iloc[:,1].values.reshape(-1,1)

lr.fit(x,y)
y_head = lr.predict(x)

from sklearn.metrics import r2_score
print("r2 : " , r2_score(y,y_head))

