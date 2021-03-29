import ID3
import numpy as np
import pandas as pd
import math
import random as rnd


def euclid_distance(centroid1, centroid2):
    sum_val = 0
    for i in range(len(centroid2)):
        sum_val = sum_val + ((centroid1[i] - centroid2[i]) ** 2)
    return math.sqrt(sum_val)


def classify_by_k_trees(obj, trees, features, k):
    obj_centroid = obj[1:]
    trees_and_euc = []
    for tree, centroid in trees:
        euc_dist = euclid_distance(obj_centroid, centroid)
        trees_and_euc.append((tree, euc_dist))

    sorted_trees_by_euc = sorted(trees_and_euc, key=lambda x: x[1])
    best_k_trees = sorted_trees_by_euc[:k]
    classifies_weights = []
    sum_euc_dist = sum(val for _, val in best_k_trees)
    for dt_tree, euc in best_k_trees:
        classify = ID3.Classifier(obj, dt_tree, features)
        classifies_weights.append((classify, (1 / euc) / (1 / sum_euc_dist)))

    sum_classification = 0
    for c, w in classifies_weights:
        if c is True:
            sum_classification = sum_classification + w
        else:
            sum_classification = sum_classification - w

    if sum_classification > 0:
        return True
    return False


def learn_n_trees(examples, n, p):
    size = len(examples.examples_matrix)
    choose = int(size * p)
    n_trees = []
    for n_i in range(n):
        random_choose = rnd.sample(range(size), choose)
        target_examples = ID3.Examples(examples.attributes, examples.examples_matrix[random_choose])
        features, centroid = [], []
        for fe in list(target_examples.attributes):
            if fe != "diagnosis":
                f_in = target_examples.attributes.index(fe)
                fe_vals = target_examples.examples_matrix[:, f_in]
                centroid_value = sum(fe_vals) / len(target_examples.examples_matrix)
                centroid.append(centroid_value)
                features.append(ID3.Feature(fe_vals, fe))
        decision_tree = ID3.ID3(target_examples, features)
        n_trees.append((decision_tree, centroid))
    return n_trees


if __name__ == '__main__':
    data_set = pd.read_csv('train.csv')
    training_set = ID3.Examples(list(data_set), data_set)
    features_list = []
    for ff in list(training_set.attributes):
        if ff != "diagnosis":
            f_i = training_set.attributes.index(ff)
            vals = training_set.examples_matrix[:, f_i]
            features_list.append(ID3.Feature(vals, ff))

    tests = np.array(pd.read_csv('test.csv'))
    # after some experiments, turns out these values are the ones that give good improvement
    N = 30
    K = 20
    p_value = 0.3

    n_learned = learn_n_trees(training_set, N, p_value)
    counter, curr = 0, None

    # part 6.1:
    for persona in tests:
        curr = ID3.CheckState(persona)
        if classify_by_k_trees(persona, n_learned, training_set.attributes, K) == curr:
            counter = counter + 1
    print(counter / len(tests))
