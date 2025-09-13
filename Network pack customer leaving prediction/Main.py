# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
# import pandas as pd
# import os
# import numpy as np
# from sklearn.model_selection import StratifiedShuffleSplit
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# import joblib

# # 0. Ai model file declaration
# MODEL_FILE = "model.pkl"
# PIPELINE_FILE = "PIPELINE.PKL"

# # 1. Function: Build Preprocessor
# def make_preprocessor():
#     binary_features = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]

#     multi_features = [
#         "Contract", "PaymentMethod", "MultipleLines", "InternetService",
#         "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
#         "StreamingTV", "StreamingMovies"
#     ]

#     numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
#     numeric_pass = ["SeniorCitizen"]

#     preprocessor = ColumnTransformer(
#         transformers=[
#             ("binary", OneHotEncoder(drop="if_binary"), binary_features),
#             ("multi", OneHotEncoder(handle_unknown="ignore"), multi_features),
#             ("num", StandardScaler(), numeric_features),
#             ("pass", "passthrough", numeric_pass)
#         ]
#     )

#     pipe = Pipeline(steps=[("preprocessor", preprocessor)])
#     return pipe
# if not os.path.exists(MODEL_FILE):
#     # Training the Data
#     # 2. Load & Clean Data
#     data = pd.read_csv("Data.csv")

#     # Fix TotalCharges
#     data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
#     data["TotalCharges"] = data["TotalCharges"].fillna(data["TotalCharges"].median())

#     # 3. Train/Test Split
#     split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
#     for train_index, test_index in split.split(data, data["Churn"]):
#         train_set = data.iloc[train_index].copy()
#         test_set = data.iloc[test_index].copy()

#     # Separateing target
#     y_train = train_set["Churn"].map({"No": 0, "Yes": 1})
#     y_test = test_set["Churn"].map({"No": 0, "Yes": 1})

#     X_train = train_set.drop(["Churn", "customerID"], axis=1)
#     X_test = test_set.drop(["Churn", "customerID"], axis=1)

#     # 4. Applying Preprocessor Pipeline
#     pipe = make_preprocessor()

#     X_train_prepared = pipe.fit_transform(X_train)
#     X_test_prepared = pipe.transform(X_test)

#     print("Train prepared shape:", X_train_prepared.shape)
#     print("Test prepared shape:", X_test_prepared.shape)

#     # 5. Converting to DataFrame with Feature Names
#     binary_names = pipe.named_steps["preprocessor"].named_transformers_["binary"].get_feature_names_out(["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"])
#     multi_names  = pipe.named_steps["preprocessor"].named_transformers_["multi"].get_feature_names_out([
#         "Contract", "PaymentMethod", "MultipleLines", "InternetService",
#         "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
#         "StreamingTV", "StreamingMovies"
#     ])
#     num_names    = ["tenure", "MonthlyCharges", "TotalCharges"]
#     pass_names   = ["SeniorCitizen"]

#     all_features = np.concatenate([binary_names, multi_names, num_names, pass_names])

#     X_train_df = pd.DataFrame(
#         X_train_prepared.toarray() if hasattr(X_train_prepared, "toarray") else X_train_prepared,
#         columns=all_features,
#         index=X_train.index
#     )

#     # 6. Train SVC
#     svc_model = SVC(probability=True, random_state=42)
#     svc_model.fit(X_train_prepared, y_train)

#     # 7. Predictions & Evaluation
#     y_pred = svc_model.predict(X_train_prepared)
#     y_proba = svc_model.predict_proba(X_train_prepared)[:, 1]

#     print("SVC Accuracy:", accuracy_score(y_train, y_pred))
#     print("SVC F1 Score:", f1_score(y_train, y_pred))
#     print("SVC ROC-AUC:", roc_auc_score(y_train, y_proba))
#     joblib.dump(svc_model, MODEL_FILE)
#     joblib.dump(pipe, PIPELINE_FILE)

# else:
#     #lets do inference 
#     model = joblib.load(MODEL_FILE)
#     Pipeline = joblib.load(PIPELINE_FILE)
#     input_data = pd.read_csv("input")
#     transformed_input = Pipeline.transform(input_data)
#     predictions = model.predict(transformed_input)
#     input_data["Churn"] = predictions
#     input_data.to_csv("output.csv", index=False)
#     print("Inference complete. Results saved to output.csv")
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib

# 0. AI model file declaration
MODEL_FILE = "model.pkl"
PIPELINE_FILE = "PIPELINE.PKL"
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
# Training / Inference
if not os.path.exists(MODEL_FILE):
    # 2. Load & Clean Data
    data = pd.read_csv("Data.csv")
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    data["TotalCharges"] = data["TotalCharges"].fillna(data["TotalCharges"].median())

    # 3. Train/Test Split
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(data, data["Churn"]):
        train_set = data.iloc[train_index].copy()
        test_set = data.iloc[test_index].copy()

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

    # 5. Convert to DataFrame (optional)
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

    # 6. Train SVC
    svc_model = SVC(probability=True, random_state=42)
    svc_model.fit(X_train_prepared, y_train)

    # 7. Predictions & Evaluation (Train Set)
    y_pred_train = svc_model.predict(X_train_prepared)
    y_proba_train = svc_model.predict_proba(X_train_prepared)[:, 1]

    print("SVC Accuracy (Train):", accuracy_score(y_train, y_pred_train))
    print("SVC F1 Score (Train):", f1_score(y_train, y_pred_train))
    print("SVC ROC-AUC (Train):", roc_auc_score(y_train, y_proba_train))

    # 8. Save Model & Pipeline
    joblib.dump(svc_model, MODEL_FILE)
    joblib.dump(pipe, PIPELINE_FILE)

    # 9. Save Test Set to CSV
    test_output = X_test.copy()
    test_output["Actual_Churn"] = y_test.values
    test_output.to_csv("input.csv", index=False)
    print("Test set saved to input.csv")

else:
    # Inference on new data
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)
    input_data = pd.read_csv("input.csv")  # replace with your input file
    transformed_input = pipeline.transform(input_data)
    predictions = model.predict(transformed_input)
    probabilities = model.predict_proba(transformed_input)[:, 1]
    input_data["Churn_Predicted"] = predictions
    input_data["Churn_Probability"] = probabilities
    input_data.to_csv("output.csv", index=False)
    print("Inference complete. Results saved to output.csv")
