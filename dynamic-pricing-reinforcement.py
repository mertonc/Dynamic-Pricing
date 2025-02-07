# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 15:07:06 2025

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
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Load the processed dataset
file_path = r"C:\Users\merto\Documents\Data Projects\processed_ecommerce_data.csv"
data = pd.read_csv(file_path)

# Convert date column to datetime
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])

# Feature Engineering - Create rolling averages for price trends
data['RollingPriceMean'] = data.groupby('StockCode')['UnitPrice'].transform(lambda x: x.rolling(5, min_periods=1).mean())

# Introduce interaction terms to balance CompetitorPrice impact
data['CompetitorPrice_Month_Interaction'] = data['CompetitorPrice'] * data['Month']
data['CompetitorPrice_Elasticity_Interaction'] = data['CompetitorPrice'] * data['PriceElasticity']

# Reduce CompetitorPrice dominance by normalizing within StockCode
data['CompetitorPrice_Normalized'] = data.groupby('StockCode')['CompetitorPrice'].transform(lambda x: (x - x.mean()) / (x.std() + 1e-5))

# Feature Selection: Use more diverse features
X = data[['Quantity', 'RollingPriceMean', 'CompetitorPrice_Normalized', 'Month', 'DayOfWeek',
          'CompetitorPrice_Month_Interaction', 'CompetitorPrice_Elasticity_Interaction']]
y = data['UnitPrice']

# Handle missing values before splitting
X.fillna(X.median(), inplace=True)
y.fillna(y.median(), inplace=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standard Scaling (Apply only to training data, then transform test data)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame for better feature tracking
X_train = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test = pd.DataFrame(X_test_scaled, columns=X.columns)

# Reinforcement Learning Environment
class PricingEnv(gym.Env):
    def __init__(self, data):
        super(PricingEnv, self).__init__()
        self.data = data.fillna(0)  # Fill missing values with zero to prevent NaN errors
        self.current_step = 0
        self.action_space = gym.spaces.Box(low=0.2, high=3.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(X_train.shape[1],), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        return self.data.iloc[self.current_step][['Quantity', 'RollingPriceMean', 'CompetitorPrice_Normalized', 'Month', 'DayOfWeek', 'CompetitorPrice_Month_Interaction', 'CompetitorPrice_Elasticity_Interaction']].values

    def step(self, action):
        self.current_step += 1
        cost = self.data.iloc[self.current_step]['UnitPrice'] * 0.6  # Assume cost is 60% of unit price
        profit = (action * self.data.iloc[self.current_step]['UnitPrice'] - cost) * self.data.iloc[self.current_step]['Quantity']
        reward = max(profit, 0)  # Ensure no negative rewards
        reward = 0 if np.isnan(reward) or np.isinf(reward) else reward
        done = self.current_step >= len(self.data) - 1
        obs = self.data.iloc[self.current_step][['Quantity', 'RollingPriceMean', 'CompetitorPrice_Normalized', 'Month', 'DayOfWeek', 'CompetitorPrice_Month_Interaction', 'CompetitorPrice_Elasticity_Interaction']].values if not done else np.zeros(self.observation_space.shape)
        obs = np.nan_to_num(obs)  # Convert any NaN values to zero
        return obs, reward, done, {}

env = DummyVecEnv([lambda: PricingEnv(data)])

# Train Reinforcement Learning Model
model = PPO('MlpPolicy', env, verbose=1, learning_rate=0.0003, gamma=0.99, ent_coef=0.05)
model.learn(total_timesteps=200000)

# Evaluate Reinforcement Learning Model
obs = env.reset()
total_rewards = []
pricing_actions = []

for _ in range(len(X_test)):
    action, _ = model.predict(obs)
    obs, rewards, done, _ = env.step(action)
    total_rewards.append(rewards[0])
    pricing_actions.append(action[0])
    if done:
        break

# Compute performance metrics
total_revenue = sum(total_rewards)
avg_price_adjustment = np.mean(pricing_actions)

# Print RL Model Evaluation Results
print(f"Total Revenue Earned by RL Model: {total_revenue:.2f}")
print(f"Average Price Adjustment by RL Model: {avg_price_adjustment:.4f}")

# Enhanced Revenue Over Time Visualization
plt.figure(figsize=(12, 6))

# Compute rolling average to smooth fluctuations
rolling_avg_rewards = pd.Series(total_rewards).rolling(window=10).mean()

# Plot original revenue trend
sns.lineplot(x=range(len(total_rewards)), y=total_rewards, label='Revenue per Step', color='blue', alpha=0.5)

# Plot rolling average revenue trend
sns.lineplot(x=range(len(rolling_avg_rewards)), y=rolling_avg_rewards, label='Rolling Avg Revenue', color='red', linestyle='dashed')

# Plot cumulative revenue to show total earnings
total_cumulative_rewards = np.cumsum(total_rewards)
sns.lineplot(x=range(len(total_cumulative_rewards)), y=total_cumulative_rewards, label='Cumulative Revenue', color='green')

plt.xlabel('Time Step')
plt.ylabel('Revenue')
plt.title('Enhanced Revenue Performance Over Time by RL Model')
plt.legend()
plt.grid()
plt.show()
plt.figure(figsize=(10, 6))
plt.plot(total_rewards, marker='o', linestyle='-', color='blue', alpha=0.7)
plt.xlabel('Time Step')
plt.ylabel('Revenue Earned')
plt.title('Revenue Earned Over Time by RL Model')
plt.grid()
plt.show()
obs = env.reset()
for _ in range(10):
    action, _ = model.predict(obs)
    obs, rewards, done, _ = env.step(action)
    if done:
        break

# Feature Importance Analysis
importances = model.policy.mlp_extractor.policy_net[0].weight.abs().mean(dim=1).detach().numpy()

# Ensure the length of importances matches the number of features
if len(importances) != len(X.columns):
    importances = np.resize(importances, len(X.columns))
feature_names = X.columns

# Display feature importances
feature_importance_df = pd.DataFrame({'Feature': ['Total Quantity', 'Rolling Avg. Price', 'Normalized Competitor Price', 'Month', 'Day of Week', 'Competitor Price × Month', 'Competitor Price × Elasticity'], 'Importance': importances}).sort_values(by='Importance', ascending=False)

# User-friendly data visualization with seaborn
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='Blues_r')
plt.xlabel('Importance Scores')
plt.ylabel('Feature Names', fontsize=12, labelpad=10)
plt.title('Updated Feature Importance in Dynamic Pricing Model')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()

# Print final model evaluation metrics
rf_model = RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_split=5, max_features='sqrt', random_state=42)
rf_model.fit(X_train, y_train)
rmse = mean_squared_error(y_test, rf_model.predict(X_test), squared=False)
print(f"Final RMSE: {rmse:.4f}")
r2 = r2_score(y_test, rf_model.predict(X_test))
print(f"Final R² Score: {r2:.4f}")