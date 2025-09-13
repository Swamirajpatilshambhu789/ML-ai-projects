import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

data = pd.read_csv('Orignal_Data.csv', encoding='latin1')  # <-- changed line

# Combine scores into a single string for stratification
split = StratifiedShuffleSplit(n_splits=1, test_size=0.22, random_state=42)
for train_index, test_index in split.split(data, data['Home_score'], data['Away_score']):
    strat_train_set = data.loc[train_index]
    strat_test_set = data.loc[test_index]

strat_test_set.to_csv('test_set.csv', index=False)