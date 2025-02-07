# Dynamic-Pricing

This project finds the best features that can best determine dynamic pricing based off of e-commerce data from (https://www.kaggle.com/datasets/carrie1/ecommerce-data/code?datasetId=1985&sortBy=voteCount) and data from simulating competitor features. The model used to find the most impactful feature is random forest regressor and reinforcement learning. 

Results show that Random Forest Regressor is a better model for determining dynmaic pricing compared to Reinforcement Learning. 



The random forest model created showed that the feature Competitor Price x Month has the highest importance score at around 3.7. 

![image](https://github.com/user-attachments/assets/1f92aa48-a2b0-45bb-8b52-02571376d667)

The RMSE and R² score shows that the model is a good predictor. The high R² can be a result of overfitting of the data. For future analysis using non-simulated competitor data can create a more compelling comparison. 

RMSE: 9.3098
R² Score: 0.9354
