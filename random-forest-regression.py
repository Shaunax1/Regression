import pandas as pd 
import matplotlib.pyplot as plt

data.columns =["id","sepal_length","sepal_width","petal_length","petal_width","species"]
data.drop(["id"],axis=1)
data["species"].replace(["Iris-setosa" , "Iris-versicolor" , "Iris-virginica"], [1,2,3])
data.isnull().sum()

x = data.iloc[:,0].values.reshape(-1,1)
y = data.iloc[:,2].values.reshape(-1,1)

plt.scatter(x,y, color="red")
plt.xlabel("sepal-length")
plt.ylabel("petal-length")
plt.show()

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=150,random_state=42)
x = data.iloc[:,0].values.reshape(-1,1)
y = data.iloc[:,2].values
rf.fit(x,y)

y_head = rf.predict(x)

plt.scatter(x,y,color="red")
plt.plot(x,y_head,color="green")
plt.xlabel("sepal")
plt.ylabel("petal")
plt.show()

from sklearn.metrics import r2_score
print("R^2 : ",r2_score(y,y_head))
