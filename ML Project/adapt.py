# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# for calculating mean_squared error
from sklearn.metrics import mean_squared_error

# Importing Linear Regression
from sklearn.linear_model import LinearRegression

# importing libraries for polynomial transform
from sklearn.preprocessing import PolynomialFeatures

# importing libraries for polynomial transform
from sklearn.preprocessing import PolynomialFeatures

# for creating pipeline
from sklearn.pipeline import Pipeline

#Reads the excel file as a dataframe
workbook = pd.read_csv('2914069.csv', usecols = ['DATE', 'TEMP'], nrows=1200)
workbook.head()

y = list(workbook['TEMP'])

x = []
for num in range(1200):
    x.append(num + 1)

x = np.array(x)
y = np.array(y)

# Training Model
lm=LinearRegression()
lm.fit(x.reshape(-1,1),y.reshape(-1,1))
y_pred=lm.predict(x.reshape(-1,1))

# creating pipeline and fitting it on data
Input=[('polynomial',PolynomialFeatures(degree=5)),('modal',LinearRegression())]
pipe=Pipeline(Input)
pipe.fit(x.reshape(-1,1),y.reshape(-1,1))
poly_pred=pipe.predict(x.reshape(-1,1))
#sorting predicted values with respect to predictor
sorted_zip = sorted(zip(x,poly_pred))
x_poly, poly_pred = zip(*sorted_zip)

#plotting predictions
plt.figure(figsize=(10,6))
plt.scatter(x,y,s=15)
plt.plot(x,y_pred,color='r',label='Linear Regression')
plt.plot(x_poly,poly_pred,color='g',label='Polynomial Regression')
plt.xlabel('Time',fontsize=16)
plt.ylabel('Temp',fontsize=16)
plt.legend()
plt.show()

print('RMSE for Polynomial Regression=>',np.sqrt(mean_squared_error(y,poly_pred)))
