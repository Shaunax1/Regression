import pandas as pd 
import matplotlib.pyplot as plt

data = pd.read_csv("/kaggle/input/iris/Iris.csv")
data.columns =["id","sepal_length","sepal_width","petal_length","petal_width","species"]
data.drop(["id"],axis=1)
data["species"].replace(["Iris-setosa" , "Iris-versicolor" , "Iris-virginica"], [1,2,3])
data.isnull().sum()


from sklearn.linear_model import LinearRegression
mlr = LinearRegression()

x = data.iloc[:,[0,1]].values
y = data.iloc[:,4]

mlr.fit(x,y)
y_head2 = mlr.predict(x)

from sklearn.metrics import r2_score
print("r2 : " , r2_score(y,y_head2))



