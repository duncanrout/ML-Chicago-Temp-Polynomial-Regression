import numpy
import pandas as pd
import matplotlib.pyplot as plt

#Reads the excel file as a dataframe
workbook = pd.read_csv('2914069.csv', usecols = ['DATE', 'TEMP'], nrows=1200)
workbook.head()

y = list(workbook['TEMP'])

x = []
for num in range(1200):
    x.append(num + 1)

mymodel = numpy.poly1d(numpy.polyfit(x, y, 9))

z = []
for i in range(1200):
    z.append(mymodel(i))

#plotting predictions
plt.figure(figsize=(10,6))
plt.scatter(x,y,s=15)
plt.scatter(x,z,s=15,color='g')
plt.xlabel('Predictor',fontsize=16)
plt.ylabel('Target',fontsize=16)
plt.legend()
plt.show()
