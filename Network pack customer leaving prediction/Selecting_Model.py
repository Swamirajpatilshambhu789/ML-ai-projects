#
# Imports
# Linear Models
from sklearn.linear_model import LogisticRegression, RidgeClassifier

# Tree-Based Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier

# Support Vector Machines
from sklearn.svm import SVC, LinearSVC

# Neighbors
from sklearn.neighbors import KNeighborsClassifier

# Naive Bayes
from sklearn.naive_bayes import GaussianNB, BernoulliNB

# Metrics & Utilities
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

import pandas as pd
import numpy as np

# 0. File Declarations
MODEL_FILE = "model.pkl"
PIPELINE_FILE = "PIPELINE.pkl"

# 1. Function: Build Preprocessor
def make_preprocessor():
    binary_features = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]

    multi_features = [
        "Contract", "PaymentMethod", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies"
    ]

    numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
    numeric_pass = ["SeniorCitizen"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("binary", OneHotEncoder(drop="if_binary"), binary_features),
            ("multi", OneHotEncoder(handle_unknown="ignore"), multi_features),
            ("num", StandardScaler(), numeric_features),
            ("pass", "passthrough", numeric_pass)
        ]
    )

    pipe = Pipeline(steps=[("preprocessor", preprocessor)])
    return pipe

# 2. Load & Clean Data
data = pd.read_csv("Data.csv")

# Fix TotalCharges
data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
data["TotalCharges"] = data["TotalCharges"].fillna(data["TotalCharges"].median())

# 3. Train/Test Split
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(data, data["Churn"]):
    train_set = data.iloc[train_index].copy()
    test_set = data.iloc[test_index].copy()

# Separate target
y_train = train_set["Churn"].map({"No": 0, "Yes": 1})
y_test = test_set["Churn"].map({"No": 0, "Yes": 1})

X_train = train_set.drop(["Churn", "customerID"], axis=1)
X_test = test_set.drop(["Churn", "customerID"], axis=1)

# 4. Apply Preprocessor Pipeline
pipe = make_preprocessor()

X_train_prepared = pipe.fit_transform(X_train)
X_test_prepared = pipe.transform(X_test)

print("Train prepared shape:", X_train_prepared.shape)
print("Test prepared shape:", X_test_prepared.shape)

# 5. Convert to DataFrame (optional, for inspection)
binary_names = pipe.named_steps["preprocessor"].named_transformers_["binary"].get_feature_names_out(
    ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
)
multi_names = pipe.named_steps["preprocessor"].named_transformers_["multi"].get_feature_names_out([
    "Contract", "PaymentMethod", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies"
])
num_names = ["tenure", "MonthlyCharges", "TotalCharges"]
pass_names = ["SeniorCitizen"]

all_features = np.concatenate([binary_names, multi_names, num_names, pass_names])

X_train_df = pd.DataFrame(
    X_train_prepared.toarray() if hasattr(X_train_prepared, "toarray") else X_train_prepared,
    columns=all_features,
    index=X_train.index
)

print(X_train_df.head())

# List of models to test
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "RidgeClassifier": RidgeClassifier(random_state=42),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    "ExtraTrees": ExtraTreesClassifier(n_estimators=200, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
    "SVC": SVC(probability=True, random_state=42),
    "LinearSVC": LinearSVC(max_iter=5000, random_state=42),
    "KNeighbors": KNeighborsClassifier(),
    "GaussianNB": GaussianNB(),
    "BernoulliNB": BernoulliNB()
    # XGBClassifier, LGBMClassifier, CatBoostClassifier can be added if installed
}

# Evaluate models using 5-fold cross-validation
for name, model in models.items():
    scores = cross_val_score(model, X_train_prepared, y_train, cv=5, scoring="accuracy")
    print(f"{name} Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
