import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import random
import pickle

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
seed_everything(42)

def read_signals(filename):
    samples_count = 5000

    c = ['name', 'x', 'y']
    for i in range(0, samples_count):
        c.append(f'v{i}')
    c = c + ['cluster', 'p0', 'p1', 'p2', 'p3']
    types = {col_name: int for col_name in ['name', 'cluster', 'p0', 'p1', 'p2', 'p3']}
    types.update({col_name: np.float32 for col_name in ['x', 'y', *[f'v{i}' for i in range(0, samples_count)]]})
    
    df = pd.read_csv(filename, names=c, dtype=types)
    df = df.set_index('name', drop=True)

    return df


def write_signals(df, filename):
    df.to_csv(filename, header=False)



data = read_signals('../data/signals.csv')
data.head()

train_df = data[data.cluster != -1]
signal_columns = [f'v{j}' for j in range(0, 5000)]
train_df.head()

from sktime.classification.interval_based import TimeSeriesForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

classifier = TimeSeriesForestClassifier(random_state=42)

scores = cross_val_score(classifier, train_df[signal_columns].values, train_df.cluster.values, cv=cv, verbose=2, n_jobs=-1)

classifier.fit(train_df[signal_columns].values, train_df.cluster.values)

data['cluster'] = classifier.predict(data[signal_columns].values)

with open('../data/classifier_model', 'wb') as f:
  pickle.dump(classifier, f)

# plt.scatter(data['x'], data['y'] , c=data.cluster, cmap='Dark2')
# plt.show()

from sklearn.linear_model import Ridge
models = {}
for i, tmp_df in data.groupby(['cluster']):
  for j in range(4):
    X = tmp_df[signal_columns].values
    X = np.hstack([X, X[:, 1:] - X[:, :-1], X[:, 12:] - X[:, :-12], X[:, 30:] - X[:, :-30]])
    Y = tmp_df[f'p{j}'].values
    train_index = np.where(Y != -1)
    test_index = np.where(Y == -1)
    model = Ridge(random_state=42)
    model.fit(X[train_index], Y[train_index])
    Y[test_index] = model.predict(X[test_index])
    models[f'{i}_p{j}'] = model
    data.loc[data.cluster == i, f'p{j}'] = Y

with open('../data/ridge_models', 'wb') as f:
  pickle.dump(models, f)
  

write_signals(data, '../data/result.csv')


