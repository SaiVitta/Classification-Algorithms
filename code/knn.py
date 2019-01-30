import numpy as np
import math
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import KFold
import pandas as pd
import operator
import sys

def readData(filePath):
    data_set = []
    labels_set = []
    with open(filePath) as file:
        for line in file:
            items = line.split('\t')
            res_line = [val for i, val in enumerate(items[:-1])]
            label_val = float(items[-1].replace('\n', ''))
            labels_set.append(label_val)
            data_set.append((res_line))

    return data_set, labels_set


def eucDist(tst_pt, trn_pt):
    distance = 0
    i = 0
    while i< len(tst_pt):
        distance += pow((tst_pt[i] - trn_pt[i]), 2)
        i += 1
    return math.sqrt(distance)


def convCatData(data_set):
    dict_vect = DictVectorizer(sparse=False)
    data_frm = pd.DataFrame(data_set).convert_objects(convert_numeric=True)
    converted_data_set = dict_vect.fit_transform(data_frm.to_dict(orient='records'))
    return converted_data_set

def extractNeighbors(tst_pt, trn_dataset, trn_data_label, k):
    dists = []

    i = 0
    while i < len(trn_dataset):
        eucl_dist = eucDist(tst_pt, trn_dataset[i])
        dists.append((trn_dataset[i], trn_data_label[i], eucl_dist))
        i += 1
    dists.sort(key=operator.itemgetter(2))
    neighbors = []
    for i in range(k):
        neighbors.append((dists[i][0], dists[i][1]))
    return neighbors

def getClass(neighbors):
    votes = {}
    for i in range(len(neighbors)):
        data, response = neighbors[i]
        if response in votes:
            votes[response] += 1
        else:
            votes[response] = 1
    votes_sorted = sorted(votes.items(), key=operator.itemgetter(1), reverse=True)
    return votes_sorted[0][0]

def runKNN(train_data_set,train_data_label,test_data_set,test_data_label,k):
    predict_label=[]
    for i in range(len(test_data_set)):
        neighbors_list = extractNeighbors(test_data_set[i], train_data_set,train_data_label, k)
        predict_label.append(getClass(neighbors_list))

    test_label = test_data_label
    true_positive, false_negative, false_positive, true_negative = 0, 0 , 0 , 0

    for i in range(len(test_data_set)):
        if int(test_label[i]) is 0 and int(predict_label[i]) is 1:
            false_positive += 1
        elif int(test_label[i]) is 1 and int(predict_label[i]) is 1:
            true_positive += 1
        elif int(test_label[i]) is 0 and int(predict_label[i]) is 0:
            true_negative += 1
        elif int(test_label[i]) is 1 and int(predict_label[i]) is 0:
            false_negative += 1
    accuracy = ((true_positive + true_negative)/float(len(test_data_set))) * 100.0
    if float(true_positive + false_negative) != 0:
        precision = (true_positive/float(true_positive + false_positive)) * 100.0
        recall = (true_positive/float(true_positive + false_negative)) * 100.0
    else:
        precision = 0
        recall = 0
    if float(2 *(true_positive)) != 0:
        f1 = ((2*(true_positive))/float((2 *(true_positive)) + false_negative + false_positive)) * 100.0
    else:
        f1 = 0
    return accuracy, precision, recall, f1

def main():
    demoMode = int(sys.argv[1])
    k = int(sys.argv[2])
    train_file_path = sys.argv[3]


    if demoMode == 0:
        data_set, data_label = readData(train_file_path)

        k_fold = 10
        iterator = 0
        accuracy_total, precision_total, recall_total, f1_total = 0, 0 , 0 , 0
        kf = KFold(n_splits=k_fold, shuffle=False)

        for train_index, test_index in kf.split(data_set):
            iterator += 1
            train_data_set, test_data_set = data_set[train_index[0]:train_index[-1]+1], data_set[test_index[0]: test_index[-1]+1]
            train_data_label, test_data_label = data_label[train_index[0]:train_index[-1]+1], data_label[test_index[0]: test_index[-1]+1]

            train_data_set = convCatData(train_data_set)
            test_data_set = convCatData(test_data_set)

            accuracy, precision, recall, f1 = runKNN(train_data_set,train_data_label,test_data_set,test_data_label, k)

            accuracy_total += accuracy
            precision_total += precision
            recall_total += recall
            f1_total += f1

            avg_accuracy = accuracy_total/k_fold
            avg_precision = precision_total/k_fold
            avg_recall = recall_total/k_fold
            avg_f1 = f1_total/k_fold

        print("Average Accuracy is:", avg_accuracy)
        print("Average precision is:", avg_precision)
        print("Average recall is:", avg_recall)
        print("Average f1 is:", avg_f1)

    else:
        training_set = np.loadtxt("project3_dataset3_train.txt")
        train_data_label = training_set[:, -1]
        train_data_set = np.delete(training_set, -1, axis = 1)

        testing_set = np.loadtxt("project3_dataset3_test.txt")
        test_data_label = testing_set[:,-1]
        test_data_set = np.delete(testing_set, -1, axis = 1)

        train_data_set = convCatData(train_data_set)
        test_data_set = convCatData(test_data_set)

        accuracy, precision, recall, f1 = runKNN(train_data_set,train_data_label,test_data_set,test_data_label, k)

        print("Performance For demo data:")
        print("Accuracy is:", accuracy)
        print("precision is:", precision)
        print("recall is:", recall)
        print("f1 is:", f1)


if __name__ == "__main__":
    main()
