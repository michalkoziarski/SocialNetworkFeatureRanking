import os
import json
import argparse

from datasets import load
from feature_selection import select, score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler


parser = argparse.ArgumentParser()
parser.add_argument('-name')
parser.add_argument('-iteration', type=int)

args = vars(parser.parse_args())

print('Running iteration #%d for dataset %s...' % (args['iteration'], args['name']))

RESULTS_PATH = os.path.join(os.path.dirname(__file__), 'results')
TRIAL_PATH = os.path.join(RESULTS_PATH, args['name'])

for path in [RESULTS_PATH, TRIAL_PATH]:
    if not os.path.exists(path):
        os.mkdir(path)

partitions = load(args['name'])

for i in range(5):
    partition = partitions[i]

    for j in range(2):
        X_train, y_train = partition[j % 2]
        X_test, y_test = partition[(j + 1) % 2]

        scaler = MinMaxScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        mask = select(X_train, y_train, verbose=True)

        base_score = score(X_train, y_train, X_test, y_test, RandomForestClassifier())
        selection_score = score(X_train[:, mask], y_train, X_test[:, mask], y_test, RandomForestClassifier())
        features = []

        for k in range(len(mask)):
            if mask[k]:
                features.append(k)

        result = {
            'features': features,
            'base_score': base_score,
            'selection_score': selection_score
        }

        file_name = '%s.iteration.%d.partition.%d.fold.%d.json' % (args['name'], args['iteration'], i + 1, j + 1)

        with open(os.path.join(TRIAL_PATH, file_name), 'w') as f:
            json.dump(result, f)
