import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Load Excel file
df = pd.read_excel("Compare.xlsx", engine="openpyxl")

print(df.head())  # Preview first few rows

# Assuming columns: [Actual, Predicted, Probability]
y_true = df.iloc[:, 0]   # First column
y_pred = df.iloc[:, 1]   # Second column
y_proba = df.iloc[:, 2]  # Third column

# Metrics
print("Accuracy:", accuracy_score(y_true, y_pred))
print("F1 Score:", f1_score(y_true, y_pred))
print("ROC-AUC:", roc_auc_score(y_true, y_proba))
