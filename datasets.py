import os
import re
import pandas as pd

from sklearn.model_selection import StratifiedKFold


EVENT_TYPES = ['continuing', 'dissolving', 'growing', 'merging', 'shrinking', 'splitting']
ORIGINAL_PATH = os.path.join(os.path.dirname(__file__), 'data', 'original')
FOLDS_PATH = os.path.join(os.path.dirname(__file__), 'data', 'folds')


def partition(name):
    os.mkdir(os.path.join(FOLDS_PATH, name))

    df = pd.read_csv(os.path.join(ORIGINAL_PATH, name + '.csv'))
    df = df.select(lambda x: not re.search('IlhanActiveness', x), axis=1)
    df = df.drop_duplicates()

    if name.endswith('1state'):
        columns = list(df.columns)[1:] + list(df.columns)[0:1]
        df = df[columns]

    columns = list(df.columns)
    columns[-1] = 'event_type'
    df.columns = columns

    df = df[df.drop('event_type', 1).duplicated() == False]

    matrix = df.as_matrix()
    X, y = matrix[:, :-1], matrix[:, -1]

    for i in range(5):
        skf = StratifiedKFold(n_splits=2, shuffle=True)
        skf.get_n_splits(X, y)

        for train_index, test_index in skf.split(X, y):
            train_set = df.iloc[train_index]
            test_set = df.iloc[test_index]

            dfs = [train_set, test_set]

            for j in range(2):
                file_name = name + '.partition.' + str(i + 1) + '.fold.' + str(j + 1) + '.csv'
                path = os.path.join(FOLDS_PATH, name, file_name)
                dfs[j].to_csv(path, index=False)


def load(name):
    path = os.path.join(FOLDS_PATH, name)
    partitions = []

    for i in range(1, 6):
        partition_name = name + '.partition.%d' % i
        folds = []

        for j in range(1, 3):
            fold_name = partition_name + '.fold.%d' % j + '.csv'

            df = pd.read_csv(os.path.join(path, fold_name))
            matrix = df.as_matrix()
            X, y = matrix[:, :-1], matrix[:, -1]

            for k in range(len(y)):
                y[k] = EVENT_TYPES.index(y[k])

            folds.append([X, y.astype(int)])

        partitions.append(folds)

    return partitions
