import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#Reads the excel file as a dataframe
workbook = pd.read_csv('2914069.csv', usecols = ['DATE', 'TEMP'], nrows=600)
workbook.head()

date = list(workbook['DATE'])
temp = list(workbook['TEMP'])

plt.figure(figsize=(10,10))
plt.xticks(rotation=45)
sns.scatterplot(date,temp)
plt.title("Average Temperatures Chicago")
plt.show()
