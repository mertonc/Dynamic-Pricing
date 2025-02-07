# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 16:31:10 2025

@author: merto
"""

import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 

file_path = r"C:\Users\merto\Documents\Data Projects\ecommerce_data.csv"
data = pd.read_csv(file_path, encoding='ISO-8859-1')

pd.set_option('display.max_columns', None)

print(data.head())
print(data.info())
print(data.isnull().sum())

#Drop rows with missing values 
data = data.dropna(subset=['CustomerID', 'Description'])
print(data.isnull().sum())


#Drop duplicate values 
data = data.drop_duplicates()

#Remove invalid transactions
data = data[(data['Quantity'] > 0) & (data['UnitPrice'] > 0)]

#create total price of value per row 
data['TotalPrice'] = data['Quantity'] * data['UnitPrice']

#Convert InvoiceDate to datetime 
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])

#Extract time-based 
data['Year'] = data['InvoiceDate'].dt.year
data['Month'] = data['InvoiceDate'].dt.month
data['Day'] = data['InvoiceDate'].dt.day 
data['Hour'] = data['InvoiceDate'].dt.hour 
data['DayOfWeek'] = data['InvoiceDate'].dt.dayofweek 

#Group by StockCode and date (item)
daily_data = data.groupby(['StockCode', 'InvoiceDate']).agg({
    'Quantity': 'sum',
    'TotalPrice': 'sum',
    'UnitPrice': 'mean'
}).reset_index()

#Display daily data
print(daily_data.head())

#Simulate competitor  
daily_data['CompetitorPrice'] = daily_data['UnitPrice'] * (1 + np.random.uniform(-0.2, 0.2, size=len(daily_data)))

#calculate demand elasticity 
daily_data['PriceElasticity'] = daily_data['Quantity']/daily_data['UnitPrice']

#Convert StockCode to Categorical
daily_data['StockCode']=daily_data['StockCode'].astype('category').cat.codes
#correlation matrix 
correlation_matrix = daily_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show() 

#plot sales over time 
daily_data['InvoiceDate'] = pd.to_datetime(daily_data['InvoiceDate'])
daily_sales = daily_data.groupby(daily_data['InvoiceDate'].dt.date)['TotalPrice'].sum()

plt.plot(daily_sales.index, daily_sales.values)
plt.title('Daily Sales Trend')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)
plt.show()

#save cleaned dataset
daily_data.to_csv(r"C:\Users\merto\Documents\Data Projects\processed_ecommerce_data.csv", index=False)
print("Processed dataset saved!")
















