import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score

data = pd.read_csv('Data.csv')
X = data.drop(columns=['Result'])  # Features, adjust as necessary
Labels = ['Home_Team','Away_team','Home_score','Away_score','Season,Result']
Predicting_label = data['Result']  # Adjust this to your target variable
# Random Forest
# Random_Forest = RandomForestClassifier(random_state=42)
# Random_Forest_error = -cross_val_score(Random_Forest, X, Predicting_label, scoring="neg_root_mean_squared_error", cv=10)
# print("Random Forest RMSE:", pd.Series(Random_Forest_error.mean()))


# # # Gradient Boosting
# Gradient_Boosting = GradientBoostingClassifier(random_state=42)
# Gradient_Boosting_error = -cross_val_score(Gradient_Boosting, X, Predicting_label, scoring="neg_root_mean_squared_error", cv=10)
# print("Gradient_Boosting_error RMSE:", pd.Series(Gradient_Boosting_error.mean()))

# XGBoost
XGB = XGBClassifier(random_state=42)
XGB_error = -cross_val_score(XGB, X, Predicting_label, scoring="neg_root_mean_squared_error", cv=10)
print("Xgb_error RMSE:", pd.Series(XGB_error.mean()))