from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
import pandas as pd
from sklearn.model_selection import cross_val_score

data = pd.read_csv('Data.csv')

X = data.drop(columns=['Away_score', 'Home_score'])
y = data[['Away_score', 'Home_score']]

# Random Forest (supports multi-output natively)
Random_forest_regressor = RandomForestRegressor(random_state=42)
rf_scores = cross_val_score(Random_forest_regressor, X, y, cv=5, scoring='neg_root_mean_squared_error')
print("Random Forest RMSE:", -rf_scores.mean())

# Gradient Boosting (wrap with MultiOutputRegressor)
Gradient_boosting_regressor = MultiOutputRegressor(GradientBoostingRegressor(random_state=42))
gb_scores = cross_val_score(Gradient_boosting_regressor, X, y, cv=5, scoring='neg_root_mean_squared_error')
print("Gradient Boosting RMSE:", -gb_scores.mean())

# XGBoost (wrap with MultiOutputRegressor)
XGB_regressor = MultiOutputRegressor(XGBRegressor(random_state=42))
xgb_scores = cross_val_score(XGB_regressor, X, y, cv=5, scoring='neg_root_mean_squared_error')
print("XGBoost RMSE:", -xgb_scores.mean())

