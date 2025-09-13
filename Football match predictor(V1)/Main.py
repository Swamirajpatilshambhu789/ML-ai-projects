from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from category_encoders import TargetEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
import pandas as pd
import os
import joblib

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

def create_match_pipeline():
    preprocessor = ColumnTransformer(
        transformers=[
            # REMOVE 'Result' from here!
            ('season_ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ['Season']),
            ('home_target', TargetEncoder(), ['Home_Team']),
            ('away_target', TargetEncoder(), ['Away_team']),
        ],
        remainder='passthrough'
    )
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('scaler', StandardScaler())
    ])
    return pipeline

if not os.path.exists(MODEL_FILE):  
    data = pd.read_csv('Orignal_Data.csv', encoding='latin1')

    # Drop rows with any NaN values
    data = data.dropna()

    # Combine scores for stratification
    data['score_combo'] = data['Home_score'].astype(str) + '_' + data['Away_score'].astype(str)
    combo_counts = data['score_combo'].value_counts()
    data = data[data['score_combo'].isin(combo_counts[combo_counts > 1].index)].reset_index(drop=True)

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.22, random_state=42)
    for train_index, test_index in split.split(data, data['score_combo']):
        train_set = data.loc[train_index]
        test_set = data.loc[test_index]

    X_train = train_set.drop(columns=['Home_score', 'Away_score', 'score_combo', 'Result'], errors='ignore')
    y_train = train_set[['Home_score', 'Away_score']]
    X_test = test_set.drop(columns=['Home_score', 'Away_score', 'score_combo', 'Result'], errors='ignore')
    y_test = test_set[['Home_score', 'Away_score']]

    # Ensure all categorical columns are string type
    categorical_cols = ['Result', 'Season', 'Home_Team', 'Away_team']
    for col in categorical_cols:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype(str)
        if col in X_test.columns:
            X_test[col] = X_test[col].astype(str)

    pipeline = create_match_pipeline()
    X_train_enc = pipeline.fit_transform(X_train, y_train['Home_score'])
    X_test_enc = pipeline.transform(X_test)

    # Now drop 'Result' from X_train and X_test if needed (not strictly necessary after encoding)
    # X_train = X_train.drop(columns=['Result'], errors='ignore')
    # X_test = X_test.drop(columns=['Result'], errors='ignore')

    model = MultiOutputRegressor(XGBRegressor(random_state=42))
    model.fit(X_train_enc, y_train)
    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)
    print("Model is trained. Congrats!")

    score = model.score(X_test_enc, y_test)
    print(f"Test R^2 score: {score}")
else:
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)
    input_data = pd.read_csv('test_set.csv')
    input_data = input_data.drop(columns=['Result'], errors='ignore')
    X_input_enc = pipeline.transform(input_data.drop(columns=['Home_score', 'Away_score'], errors='ignore'))
    predictions = model.predict(X_input_enc)
    input_data[["Home_score", "Away_score"]] = predictions.astype(int)
    input_data["Result"] = input_data.apply(
        lambda row: "win" if row["Home_score"] > row["Away_score"] else ("lose" if row["Home_score"] < row["Away_score"] else "draw"),
        axis=1
    )
    input_data.to_csv("output.csv", index=False)
    print("Inference is complete, results saved to output.csv. Enjoy!")