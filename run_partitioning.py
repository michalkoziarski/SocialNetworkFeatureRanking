import os
import datasets


with open(os.path.join(os.path.dirname(__file__), 'datasets.txt')) as f:
    names = f.readlines()

names = [name.strip() for name in names]

for name in names:
    print('Partitioning %s...' % name)
    datasets.partition(name)
