import array
import random
import numpy as np

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from deap import algorithms
from deap import base
from deap import creator
from deap import tools


def score(X_train, y_train, X_test, y_test, classifier):
    if X_train.shape[1] == 0:
        return 0.0
    else:
        clf = clone(classifier).fit(X_train, y_train)

        return f1_score(y_test, clf.predict(X_test), average='weighted')


def select(X, y, n_generations=100, population_size=500, mutation_prob=0.02, crossover_prob=0.7, bit_flip_prob=0.05,
           tournament_size=3, weight_score=0.8, weight_n_features=0.2, verbose=False):
    n_features = X.shape[1]

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", array.array, typecode='b', fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    def weighted_random(p=0.5):
        if random.random() <= p:
            return 1
        return 0

    toolbox.register("attr_bool", weighted_random, 0.3)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n_features)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(individual):
        X_train, X_val, y_train, y_val = train_test_split(X, y)
        features = np.array([bool(x) for x in individual])
        coef_score = score(X_train[:, features], y_train, X_val[:, features], y_val, RandomForestClassifier())
        coef_n_features = float(sum(features)) / len(features)

        return weight_score * coef_score - weight_n_features * coef_n_features,

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=bit_flip_prob)
    toolbox.register("select", tools.selTournament, tournsize=tournament_size)

    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    algorithms.eaSimple(pop, toolbox, cxpb=crossover_prob, mutpb=mutation_prob, ngen=n_generations, stats=stats,
                        halloffame=hof, verbose=verbose)

    features = np.array([bool(x) for x in hof[0]])

    return features
