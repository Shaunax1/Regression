import pandas as pd 
import matplotlib.pyplot as plt

data = pd.read_csv("/kaggle/input/iris/Iris.csv")
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

from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor()
dtr.fit(x,y)
y_head = dtr.predict(x)

plt.scatter(x,y, color="red")
plt.plot(x,y_head, color="green")
plt.xlabel("sepal")
plt.ylabel("petal")
plt.show()

from sklearn.metrics import r2_score
print("r2 score : " , r2_score(y,y_head))

