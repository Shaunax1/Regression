import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("Mall_Customers.csv", sep=";")

df.head()
df.describe()

df.rename( columns = {"Annual Income (k$)" : "annual_income",
                      "Spending Score (1-100)" : "spending_score"
                      }, inplace=True)

plt.scatter(df.Age,df.annual_income,df.spending_score)
plt.xlabel("spending-score,Age")
plt.ylabel("annual_income")
plt.show()


from sklearn.linear_model import LinearRegression
multipleLinear = LinearRegression()

x = df.iloc[:,[3,4]].values
y = df.iloc[:,2]

multipleLinear.fit(x,y)

y_head = multipleLinear.predict(x)
print("b0 : " , multipleLinear.intercept_)
print("b1,b2 : " , multipleLinear.coef_)


from sklearn.metrics import r2_score
print("r^2 score : ", r2_score(y,y_head))
