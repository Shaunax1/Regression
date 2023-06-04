import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("iris.csv" , sep=";")
df.describe()



plt.scatter(df.sepal_length,df.petal_length)
plt.xlabel("sepal")
plt.ylabel("petal")
plt.show()

#%%
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,2].values.reshape(-1,1)

lr.fit(x,y)

y_head = lr.predict(x)
plt.plot(x,y_head,color="red")

from sklearn.metrics import r2_score
print("r2 : " , r2_score(y,y_head))
