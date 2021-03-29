import numpy as np
import pandas as pd
import math
import copy
import sklearn.model_selection
import matplotlib.pyplot as plot

'''
    Examples class
'''
class Examples:
    def __init__(self, names, data):
        # this holds the names of the features as a list of strings
        self.attributes = names
        # this holds a 2d array for the values inside the csv file
        self.examples_matrix = np.array(data)


'''
    Node class
        feature - holds an object of class Feature
        children - holds the 2 children for the current node
        classifier - holds which class does this node refer to (True / False)
        node_value - holds the value of the maximum information gain
'''
class Node:
    def __init__(self, feature, children, classifier, node_value):
        self.feature = feature
        self.children = children
        self.classifier = classifier
        self.node_value = node_value


'''
    Feature class
        name - holds the name of the feature as string
        values - holds a list of the values for this feature
'''
class Feature:
    def __init__(self, values, feature):
        self.name = feature
        self.values = values

    # calculates a list of arrays such that each array is a person in the csv file (both forE1 and forE2)
    def forE1(self, examples, value):
        res = []
        f_index = examples.attributes.index(self.name)
        for i in range(len(examples.examples_matrix)):
            if examples.examples_matrix[i][f_index] <= value:
                res.append(examples.examples_matrix[i])
        return res

    def forE2(self, examples, value):
        res = []
        f_index = examples.attributes.index(self.name)
        for i in range(len(examples.examples_matrix)):
            if examples.examples_matrix[i][f_index] > value:
                res.append(examples.examples_matrix[i])
        return res


'''
    returns True if number of B persons is bigger than number of M persons, False otherwise
'''
def MajorityClass(examples, p = 0.5):
    count_b, count_m = 0, 0
    for i in range(len(examples.examples_matrix)):
        if examples.examples_matrix[i][0] == 'M':
            count_m = count_m + 1
        if examples.examples_matrix[i][0] == 'B':
            count_b = count_b + 1
    if count_m / len(examples.examples_matrix) > p:
        return False
    return True


'''
    Simply returns False if the given person is diagnosed as M and True otherwise (diagnosed as B)
'''
def CheckState(person):
    state = None
    if person[0] == 'M':
        state = False
    if person[0] == 'B':
        state = True
    return state


def p(examples, c):
    count = 0
    for person in examples:
        if person[0] == c:
            count = count + 1
    return count / len(examples)


'''
    calculates the entropy value with the help of the p(examples, c) function above
'''
def Entropy(examples):
    if len(examples) == 0:
        return 0

    val_b = p(examples, 'B')
    val_m = p(examples, 'M')

    if val_m == 0:
        res = -math.log(val_b, 2) * val_b
    elif val_b == 0:
        res = -math.log(val_m, 2) * val_m
    else:
        res = -((math.log(val_m, 2) * val_m) + (math.log(val_b, 2) * val_b))

    return res


'''
    calculates the best information gain value for the given feature
'''
def InformationGain(examples, feature):
    curr_entropy = Entropy(examples.examples_matrix)
    max_v = -np.inf
    max_gain = -np.inf
    f_index = examples.attributes.index(feature.name)
    sorted_values = sorted(examples.examples_matrix[:, f_index])
    target_values = [0.5 * (sorted_values[i] + sorted_values[i + 1]) for i in range(len(sorted_values) - 1)]
    for value in target_values:
        gain = curr_entropy
        e1_node = feature.forE1(examples, value)
        e2_node = feature.forE2(examples, value)
        size = len(examples.examples_matrix)
        gain = gain - (Entropy(e1_node) * len(e1_node) / size) - (Entropy(e2_node) * len(e2_node) / size)
        if gain >= max_gain:
            max_gain = gain
            max_v = value
    return max_gain, max_v


'''
    returns the feature which gives the highest information gain value (with its information gain value)
'''
def MaxInformationGain(examples, features):
    max_v = -np.inf
    max_gain = -np.inf
    best_feature = None

    for feature in features:
        gain, value = InformationGain(examples, feature)
        if gain >= max_gain:
            max_gain = gain
            max_v = value
            best_feature = feature
    return best_feature, max_v


'''
    Top down induction decision tree algorithm
'''
def TopDownInductionDT(examples, features, default, select_feature, m_value, p):
    flag = 1
    if len(examples.examples_matrix) < m_value:
        return Node(None, [], default, None)
    if len(examples.examples_matrix) == 0:
        return Node(None, [], default, None)

    c = MajorityClass(examples, p)

    for person in examples.examples_matrix:
        res = CheckState(person)
        if res != c:
            flag = 0
            break

    if flag == 1 or len(features) == 0:
        return Node(None, [], c, None)

    f, node_val = select_feature(examples, features)
    #features.remove(f)

    sub_trees = [(False, TopDownInductionDT(Examples(examples.attributes, f.forE1(examples, node_val)), features,
                                            c, select_feature, m_value, p)),
                 (True, TopDownInductionDT(Examples(examples.attributes, f.forE2(examples, node_val)), features,
                                           c, select_feature, m_value, p))]
    return Node(f, sub_trees, c, node_val)


'''
    the classifier function 
        - given an object (obj) it decides if it's True or False (B or M) according to the given decision tree
'''
def Classifier(obj, decision_tree, features):
    if len(decision_tree.children) == 0:
        return decision_tree.classifier
    for (value, sub_tree) in decision_tree.children:
        f_index = features.index(decision_tree.feature.name)
        if obj[f_index] <= decision_tree.node_value:
            res = False
        else:
            res = True
        if res == value:
            return Classifier(obj, sub_tree, features)


'''
    ID3 algorithm from the lecture
    m_value param is for part 3 and it's default value is set to 0 (in case it is not used)
'''
def ID3(examples, features, m_value = 0, p = 0.0):
    c = MajorityClass(examples, p)
    return TopDownInductionDT(examples, features, c, MaxInformationGain, m_value, p)


def experiment(examples, p_values, m_value=1):
    '''
        experiment function - for part 3.3
        how to run:
            - go to the main function (that's literally below this function, where it says "if __name__ ==...")
            - scroll down where it says "# part 3.3:"
            - read the data from train.csv file and declare an object of the class Examples as the following:
                - data_set = pd.read_csv('train.csv')
                - training_set = Examples(list(data_set), data_set)
                - # or use the variable "training_set" that I've already declared
            - declare an M list of the values you want (the default I declared is [1, 10, 30, 50, 70, 100, 150])
            - call the function experiment from main as the following:
                - experiment(training_set, <M list>)
    '''
    k_folds = sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=205526205)
    results = []
    for p_value in p_values:
        losses = []
        for train_index, test_index in k_folds.split(examples.examples_matrix):
            target_examples = Examples(examples.attributes, examples.examples_matrix[train_index])
            features = []
            for fe in list(target_examples.attributes):
                if fe != "diagnosis":
                    f_in = target_examples.attributes.index(fe)
                    fe_vals = target_examples.examples_matrix[:, f_in]
                    features.append(Feature(fe_vals, fe))
            target_tests = examples.examples_matrix[test_index]
            decision_tree = ID3(target_examples, features, m_value, p_value)
            loss, state, res = 0, None, 0
            for person in target_tests:
                state = CheckState(person)
                copy_dt = copy.deepcopy(decision_tree)
                copy_training = copy.deepcopy(target_examples)
                if not Classifier(person, copy_dt, copy_training.attributes) == state:
                    if state is False:
                        loss = loss + 1
                    else:
                        loss = loss + 0.1
                res = loss / len(target_tests)
            losses.append(res)
        results.append((p_value, sum(losses) / len(losses)))

    min_loss = np.inf
    p_ret = 0
    for p_val, loss_val in results:
        if loss_val < min_loss:
            min_loss = loss_val
            p_ret = p_val

    return p_ret, min_loss


def after_tuning_loss(examples, features, tests_set, m_value, p_val):
    loss, curr = 0, None
    full_decision_tree = ID3(examples, features, m_value, p_val)
    for person in tests_set:
        curr = CheckState(person)
        copy_decision_tree = copy.deepcopy(full_decision_tree)
        copy_training_set = copy.deepcopy(examples)
        if not Classifier(person, copy_decision_tree, copy_training_set.attributes) == curr:
            if curr is False:
                loss = loss + 1
            else:
                loss = loss + 0.1
    return loss / len(tests_set)


if __name__ == '__main__':
    data_set = pd.read_csv('train.csv')
    training_set = Examples(list(data_set), data_set)
    features_list = []
    for ff in list(training_set.attributes):
        if ff != "diagnosis":
            f_i = training_set.attributes.index(ff)
            vals = training_set.examples_matrix[:, f_i]
            features_list.append(Feature(vals, ff))

    tests = np.array(pd.read_csv('test.csv'))

    # part 4.3:
    p_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    best_m_value, p_value = 10, 0.2
    #best_p, loss_value= experiment(training_set, p_list, best_m_value)
    print(after_tuning_loss(training_set, features_list, tests, best_m_value, p_value))
