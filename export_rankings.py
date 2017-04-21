import os
import json
import numpy as np
import pandas as pd


RANKINGS_PATH = os.path.join(os.path.dirname(__file__), 'rankings')

if not os.path.exists(RANKINGS_PATH):
    os.mkdir(RANKINGS_PATH)

with open(os.path.join(os.path.dirname(__file__), 'datasets.txt')) as f:
    names = f.readlines()

names = [name.strip() for name in names]

for name in names:
    columns = list(pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'folds', name,
                                            '%s.partition.1.fold.1.csv' % name)).columns[:-1])
    results_path = os.path.join(os.path.dirname(__file__), 'results', name)

    if not os.path.exists(results_path):
        continue

    n_iterations = len(os.listdir(results_path))
    counts = np.zeros(len(columns))

    for file_name in os.listdir(results_path):
        with open(os.path.join(results_path, file_name)) as file:
            result = json.load(file)

        features = result['features']

        for feature in features:
            counts[feature] += 1

    frequencies = map(lambda x: x / float(n_iterations), counts)

    df = pd.DataFrame({'feature': columns, 'ranking': frequencies}).sort_values('ranking', ascending=False)
    df.to_csv(os.path.join(RANKINGS_PATH, '%s.csv' % name), index=False)
