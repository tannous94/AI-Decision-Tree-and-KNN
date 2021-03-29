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
def MajorityClass(examples):
    count_b, count_m = 0, 0
    for i in range(len(examples.examples_matrix)):
        if examples.examples_matrix[i][0] == 'M':
            count_m = count_m + 1
        if examples.examples_matrix[i][0] == 'B':
            count_b = count_b + 1
    if count_m > count_b:
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
def TopDownInductionDT(examples, features, default, select_feature, m_value):
    flag = 1
    if len(examples.examples_matrix) < m_value:
        return Node(None, [], default, None)
    if len(examples.examples_matrix) == 0:
        return Node(None, [], default, None)

    c = MajorityClass(examples)

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
                                            c, select_feature, m_value)),
                 (True, TopDownInductionDT(Examples(examples.attributes, f.forE2(examples, node_val)), features,
                                           c, select_feature, m_value))]
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
def ID3(examples, features, m_value = 0):
    c = MajorityClass(examples)
    return TopDownInductionDT(examples, features, c, MaxInformationGain, m_value)


def experiment(examples, m_list):
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
    for m_value in m_list:
        accuracy = []
        for train_index, test_index in k_folds.split(examples.examples_matrix):
            target_examples = Examples(examples.attributes, examples.examples_matrix[train_index])
            features = []
            for fe in list(target_examples.attributes):
                if fe != "diagnosis":
                    f_in = target_examples.attributes.index(fe)
                    fe_vals = target_examples.examples_matrix[:, f_in]
                    features.append(Feature(fe_vals, fe))
            target_tests = examples.examples_matrix[test_index]
            decision_tree = ID3(target_examples, features, m_value)
            count, state, res = 0, None, 0
            for person in target_tests:
                state = CheckState(person)
                copy_dt = copy.deepcopy(decision_tree)
                copy_training = copy.deepcopy(target_examples)
                if Classifier(person, copy_dt, copy_training.attributes) == state:
                    count = count + 1
                res = count / len(target_tests)
            accuracy.append(res)
        results.append(sum(accuracy) / len(accuracy))

    plot.plot(m_list, results, label='accuracy')
    plot.legend()
    plot.show()
    #print('\n') # instead of printing them here I have to draw a graph that shows them =D
    #for i in results:
    #    print(i)


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
    counter, curr = 0, None
    full_decision_tree = ID3(training_set, features_list)

    # part 1:
    for persona in tests:
        curr = CheckState(persona)
        copy_decision_tree = copy.deepcopy(full_decision_tree)
        copy_training_set = copy.deepcopy(training_set)
        if Classifier(persona, copy_decision_tree, copy_training_set.attributes) == curr:
            counter = counter + 1
    print(counter / len(tests))

    # part 3.3:
    m_values = [1, 10, 30, 50, 70, 100, 150]
    #experiment(training_set, m_values)

    # part 3.4:
    '''counter_2, curr_2, best_m_value = 0, None, 1
    full_dt_with_m = ID3(training_set, features_list, best_m_value)
    for persona in tests:
        curr_2 = CheckState(persona)
        copy_decision_tree = copy.deepcopy(full_dt_with_m)
        copy_training_set = copy.deepcopy(training_set)
        if Classifier(persona, copy_decision_tree, copy_training_set.attributes) == curr_2:
            counter_2 = counter_2 + 1
    print(counter_2 / len(tests))'''

    # part 4.1: calculating loss
    '''best_m_value, loss, curr_3 = 1, 0, None
    full_dt_with_m = ID3(training_set, features_list, best_m_value)
    for persona in tests:
        curr_3 = CheckState(persona)
        copy_decision_tree = copy.deepcopy(full_dt_with_m)
        copy_training_set = copy.deepcopy(training_set)
        if not Classifier(persona, copy_decision_tree, copy_training_set.attributes) == curr_3:
            if curr_3 is False:
                loss = loss + 1
            else:
                loss = loss + 0.1
    print(loss / len(tests))'''
