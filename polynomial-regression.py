import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()
data = pd.DataFrame(iris.data)

data.head()
data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

# 0 = Iris-setosa 1 = Iris-versicolor 2 = Iris-virginica
target = pd.DataFrame(iris.target)
target = target.rename(columns = {0: 'target'})
target.head()
data = pd.concat([data, target], axis = 1)

data.isnull().sum()

x = data.iloc[:,0].values.reshape(-1,1)
y = data.iloc[:,2].values.reshape(-1,1)

plt.scatter(x,y, color="red")
plt.xlabel("sepal-length")
plt.ylabel("petal-length")
plt.show()

from sklearn.preprocessing import PolynomialFeatures
plr = PolynomialFeatures(degree = 10)
x_polynomial = plr.fit_transform(x)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_polynomial,y)
y_head = lr.predict(x_polynomial)
plt.plot(x,y_head,color="blue")
plt.show()

from sklearn.metrics import r2_score
print("r2 score : ",r2_score(y,y_head))
