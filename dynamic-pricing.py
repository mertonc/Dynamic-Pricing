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

#save cleaned dataset
daily_data.to_csv(r"C:\Users\merto\Documents\Data Projects\processed_ecommerce_data.csv", index=False)
print("Processed dataset saved!")
















