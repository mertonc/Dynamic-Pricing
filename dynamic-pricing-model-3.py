# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 14:34:07 2025

@author: merto
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import uvicorn
from fastapi import FastAPI

# Load the processed dataset
file_path = r"C:\Users\merto\Documents\Data Projects\processed_ecommerce_data.csv"
data = pd.read_csv(file_path)

# Convert date to datetime
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])

# Create rolling averages for price trends
data['RollingPriceMean'] = data.groupby('StockCode')['UnitPrice'].transform(lambda x: x.rolling(5, min_periods=1).mean())

# Interaction terms to balance CompetitorPrice impact
data['CompetitorPrice_Month_Interaction'] = data['CompetitorPrice'] * data['Month']
data['CompetitorPrice_Elasticity_Interaction'] = data['CompetitorPrice'] * data['PriceElasticity']

# Reduce CompetitorPrice dominance by normalizing within StockCode
data['CompetitorPrice_Normalized'] = data.groupby('StockCode')['CompetitorPrice'].transform(lambda x: (x - x.mean()) / (x.std() + 1e-5))

# Create training and testing data
X = data[['Quantity', 'RollingPriceMean', 'CompetitorPrice_Normalized', 'Month', 'DayOfWeek',
          'CompetitorPrice_Month_Interaction', 'CompetitorPrice_Elasticity_Interaction']]
y = data['UnitPrice']

# Handle missing values before splitting
X.fillna(X.median(), inplace=True)
y.fillna(y.median(), inplace=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standard Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame 
X_train = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test = pd.DataFrame(X_test_scaled, columns=X.columns)

# Random Forest Model with Tweaked Parameters
model = RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_split=5, max_features='sqrt', random_state=42)
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

# Feature Importance Analysis
importances = model.feature_importances_
feature_names = X.columns

# Display feature importances
feature_importance_df = pd.DataFrame({'Feature': ['Total Quantity', 'Rolling Avg. Price', 
                                                  'Normalized Competitor Price', 'Month', 'Day of Week', 
                                                  'Competitor Price × Month', 'Competitor Price × Elasticity'], 'Importance': importances}).sort_values(by='Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
plt.xlabel('Importance Scores')
plt.ylabel('Feature')
plt.title('Updated Feature Importance in Dynamic Pricing Model')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()

# Print final model evaluation metrics
print(f"Final RMSE: {rmse:.4f}")
print(f"Final R² Score: {r2:.4f}")

app = FastAPI()



