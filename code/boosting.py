import numpy as np
from utility import isNum, doEncode, calculateGiniRF
from collections import Counter as c
import math
import os
import random
import sys

# The class to represent a TreeNode in a decision tree.
class TreeNode:
    def __init__(self, si = None, sv = None, splits = None, label = None):
        self.sv = sv
        self.si = si
        self.left = None
        self.right = None
        self.splits = splits
        self.label = label

# Build a decision tree.
def decisionTreeBuild(data, maxDepth, minSize, attrs):
    print(" Building Decision Tree......")
    btnode = splitBest(data, attrs)
    splitTreeNode(btnode, maxDepth, minSize, 1, attrs)
    # return the tree node
    return btnode

# function to represent a leaf node in a tree along with the label information.
def nodeLeaf(data):
    mc = c([datum[-1] for datum in data]).most_common(1)
    # a tree node which is a leaf node.
    val = mc[0][0]
    tnode = TreeNode(label = val)
    return tnode

# function to split the whole dataset based on split index and split value parameters
def dataSplit(splitLeft, splitRight, si, sv, data, category):
    length = len(data)
    if not category:
        for index in range(length):
            if data[index][si] < sv:
                sp = data[index]
                splitLeft.append(sp)
            else:
                sp = data[index]
                splitRight.append(sp)
    else:
        for index in range(length):
            if data[index][si] == sv:
                sp = data[index]
                splitLeft.append(sp)
            else:
                sp = data[index]
                splitRight.append(sp)
    return splitLeft, splitRight

# for best split in dataSet
def splitBest(data, attrs):
    minGini = sys.maxsize
    sampleCount = len(data)
    attrNum = len(data[0])
    # starting at max size
    si = sys.maxsize
    sv = sys.maxsize
    # choose one with gini index having min value
    for y in range(attrNum-1):
        if y not in attrs:
            values_seen = []
            for x in range(sampleCount):
                if data[x][y] not in values_seen:
                    values_seen.append(data[x][y])
                    t_gini = calculateGiniRF(data, y,data[x][y],False)
                    if t_gini <= minGini:
                        minGini = t_gini
                        si = y
                        sv = data[x][y]

        else:
            values_seen = []
            for x in range(sampleCount):
                if data[x][y] not in values_seen:
                    values_seen.append(data[x][y])
                    t_gini = calculateGiniRF(data, y,data[x][y],True)
                    if t_gini <= minGini:
                        minGini = t_gini
                        si = y
                        sv = data[x][y]

    splitLeft = []
    splitRight = []
    if si in attrs:
        splits = dataSplit(splitLeft, splitRight, si, sv, data, True)
    else:
        splits = dataSplit(splitLeft, splitRight, si, sv, data, False)

    tnode = TreeNode(si, sv, splits)
    return tnode

# split and build a tree till leaf node is caught
def splitTreeNode(tnode, maxDepth, minSize, curDepth, attrs):
    left, right = tnode.splits
    leftlen = len(left)
    rightlen = len(right)
    if leftlen == 0:
        tnode.left = tnode.right = nodeLeaf(right)
        return
    if rightlen == 0:
        tnode.left = tnode.right = nodeLeaf(left)
        return
    if curDepth >= maxDepth:
        tnode.left = nodeLeaf(left)
        tnode.right = nodeLeaf(right)
        return
    if leftlen <= minSize:
        tnode.left = nodeLeaf(left)
    else:
       tnode.left = splitBest(left, attrs)
       splitTreeNode(tnode.left, maxDepth, minSize, curDepth + 1, attrs)
    if rightlen <= minSize:
        tnode.right = nodeLeaf(right)
    else:
       tnode.right = splitBest(right, attrs)
       splitTreeNode(tnode.right, maxDepth, minSize, curDepth + 1, attrs)

def cross_validation(data, labels, start, end, maxDepth, minSize, attrs, testSize, sampleNum, k):
    accuracy = recall = precision = f1 = 0.0
    flag = 0
    for iter in range(k):
        print("Iteration: ", iter+1)
        testTrueLabels = labels[start:end]
        testData = data[start:end][:]
        upper = lower = 0
        misclassified, levels, testNewLabels, rts, tl, sd= ([] for i in range(6))
        trainData = np.delete(data,np.s_[start:end],0)
        indices = [i for i in range(0,len(trainData))]
        w = [1/len(trainData)]*len(trainData)
        trainLabels = np.delete(labels,np.s_[start:end])

        while len(rts)< 5:
            w = [float(i)/sum(w) for i in w]
            for index in range(len(trainData)):
                sd.append(trainData[random.choices(indices,w)].tolist()[0])
            root = decisionTreeBuild(sd, maxDepth, minSize, attrs)
            for ts in trainData:
                tl.append(classify_dt(root,ts,attrs))
            for j in range(len(trainData)):
                if trainLabels[j] != tl[j]:
                    misclassified.append(j)
                    upper += w[j]
                lower += w[j]
            error = upper/lower
            if error <= 0.5:
                level = math.sqrt(math.log((1-error)/error))
                levels.append(level)
                for j in range(len(trainData)):
                    if j in misclassified:
                        w[j] = w[j]*math.exp(level)
                    else:
                        w[j] = w[j]*math.exp(0)
                w = [float(i)/sum(w) for i in w]
                rts.append(root)

        flag = 1
        print("   Classifying test data.......")
        for ts in testData:
            labels1 = counts1 = [0,0]
            # Aggregrate the weights for each class based on what is predicted by the decision trees
            for i in range(len(rts)):
                label = int(classify_dt(rts[i],ts,attrs))
                labels1[label] += levels[i]
                counts1[label] += 1
            if counts1[0] == 0:
                testNewLabels.append(1.0)
            elif counts1[1] == 0:
                testNewLabels.append(0.0)
            elif labels1[0] > labels1[1]:
                testNewLabels.append(0.0)
            else:
                testNewLabels.append(1.0)
        # create a confusion matrix cm
        # contructing the confusion matrix based on the classification results in testNewLabels
        cm = np.zeros((2,2))
        for i in range(len(testTrueLabels)):
            if testTrueLabels[i]==testNewLabels[i]:
                if testTrueLabels[i]==1:
                    cm[0][0] += 1
                else:
                    cm[1][1] += 1
            else:
                if testTrueLabels[i]==1:
                    cm[0][1] += 1
                else:
                    cm[1][0] += 1

        total_correct = cm[0][0] + cm[1][1]
        total_predictions = cm[1][0]+cm[0][1]+cm[0][0]+cm[1][1]
        true_positives = cm[0][0]
        false_negatives = cm[0][1]
        true_negatives = cm[1][1]
        false_positives = cm[1][0]
        # Calculate the evaluation metric for kth fold using the confusion matrix.
        if total_predictions !=0:
            accuracy = accuracy + total_correct/total_predictions
        if true_positives + false_negatives != 0:
            recall = recall + true_positives/(true_positives + false_negatives)
        if true_positives + false_positives != 0:
            precision = precision + true_positives/(true_positives + false_positives)
        if (2*true_positives) + false_negatives + false_positives != 0:
            f1 = f1 + (2*true_positives)/((2*true_positives)+false_negatives + false_positives)
        if iter==8:
            start = start + testSize
            end = end + testSize+sampleNum-(10*testSize)
        else:
            start = start + testSize
            end = end + testSize
    print(" Generating evaluation metrics........")
    return accuracy, precision, recall, f1

# function to classify class label of data.
def classify_dt(tnode, test_s, attrs):
    if tnode.si not in attrs:
        if test_s[tnode.si]>=tnode.sv:
            if tnode.right.label is not None:
                return tnode.right.label
            else:
                #classify
                return classify_dt(tnode.right, test_s, attrs)
        else:
            if tnode.left.label is not None:
                return tnode.left.label
            else:
                #classify
                return classify_dt(tnode.left, test_s, attrs)

    else:
        if test_s[tnode.si] == tnode.sv:
            if tnode.left.label is not None:
                return tnode.left.label
            else:
                #classify
                return classify_dt(tnode.left, test_s, attrs)
        else:
            if tnode.right.label is not None:
                return tnode.right.label
            else:
                #classify
                return classify_dt(tnode.right, test_s, attrs)

algorithmName = sys.argv[0]
a = algorithmName.split(".")
algorithmName = a[0]
dataSet = sys.argv[1]
# reading the file set by the user as a command line argument
with open(dataSet) as tf:
    lines = [l.split('\t') for l in tf]
# Initializing various parameters required to classify and compute evaluation metrics.
minSize = -1
maxDepth = 100
sampleNum = len(lines)
attrNum = len(lines[0])
labels = [int(r[-1].rstrip("\n")) for r in lines]
data = np.zeros((sampleNum,attrNum),dtype=float)
attrs = set()
classes=np.unique(labels)
k = 10
start = 0
testSize = int(sampleNum/10)
trainSize = sampleNum - testSize
end = testSize
data,attrs = doEncode(data, attrs, lines, attrNum, sampleNum)
accuracy, precision, recall, f1 = cross_validation(data, labels, start, end, maxDepth, minSize, attrs, testSize, sampleNum, k)
print("\n")
print("Results for",algorithmName)
print("Recall is  " ,recall/10)
print("Accuracy is " ,accuracy/10)
print("Precision is " ,precision/10)
print("F1-measure is  ",f1/10)
