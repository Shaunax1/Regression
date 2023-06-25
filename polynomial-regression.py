import pandas as pd 
import matplotlib.pyplot as plt

data.columns =["id","sepal_length","sepal_width","petal_length","petal_width","species"]
data.drop(["id"],axis=1)
data["species"].replace(["Iris-setosa" , "Iris-versicolor" , "Iris-virginica"], [1,2,3])
data.isnull().sum()

from sklearn.preprocessing import PolynomialFeatures
plr = PolynomialFeatures(degree = 5)

x = data.iloc[:,0].values.reshape(-1,1)
y = data.iloc[:,2].values.reshape(-1,1)
x_polynomial = plr.fit_transform(x)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_polynomial,y)
y_head = lr.predict(x_polynomial)
plt.plot(a,y_head, color="green" , label="poly")
plt.legend()
plt.show()

from sklearn.metrics import r2_score
print("r2 score : ",r2_score(y,y_head))
