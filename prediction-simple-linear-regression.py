import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("datasets/sales_month_data.csv")
print(df)

months = df[['months']]
sales = df[['sales']]
print(months)
print(sales)

#Data train-test-split
x_train,x_test,y_train,y_test = train_test_split(months,sales,test_size=0.33,random_state=0)

#Data scaling
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)

#Model building - Simple Linear Regression
from sklearn.linear_model import LinearRegression

linearRegression = LinearRegression()
linearRegression2 = LinearRegression()
linearRegression.fit(X_train,Y_train)
linearRegression2.fit(x_train,y_train)

predict = linearRegression.predict(X_test)
#Without Standartization
predict2 = linearRegression2.predict(x_test)

#Visualization
x_train = x_train.sort_index()
y_train = y_train.sort_index()

plt.plot(x_train,y_train)
plt.plot(x_test,predict2)
plt.title('Linear Regression')
plt.xlabel('Months')
plt.ylabel('Sales')
plt.show()