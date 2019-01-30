import numpy as np
import scipy.stats as ss
import os
import sys

def isNum(number):
    try:
        # convert the number into a float value
        number = float(number)
    except:
        # if the number is not a number then return false
        return False, None
    return True, number

file_name = "project3_dataset4.txt"
file = open(file_name)
print("File selected is :",file_name)
all_lines = file.readlines()
meanStdDict = {}
main_arr = []
columns = len(all_lines[0].split("\t"))
rows = len(all_lines)
data_matrix = [[0 for x in range(columns)] for y in range(rows)]
r = 0
while r < rows:
    for c in range(columns):
        data_matrix[r][c] = all_lines[r].split("\t")[c]
        data_matrix[r][c] = data_matrix[r][c].rstrip("\n")
    r = r + 1
data_matrix = np.array(data_matrix)
leng = len(data_matrix[0])
x = 0
while x < len(data_matrix[0]):
    status, number = isNum(data_matrix[0][x])
    if x == len(data_matrix[0]) - 1:
        main_arr.append("Class")
    elif status:
        main_arr.append("Numeral")
    else:
        main_arr.append("Categorical")
    x = x + 1

def categorical(category, column_index, class_label):
    numerator = 0
    denominator = 0
    x = 0
    while x < len(data_matrix):
        if data_matrix[x][column_index] == category and data_matrix[x][len(data_matrix[0]) - 1] == class_label:
            numerator += 1
        x = x + 1
    column = data_matrix[:, len(data_matrix[0]) - 1]
    le = list(column)
    denominator = le.count(class_label)
    return numerator / denominator

# method for mean and std in dictionary
def mviddict(im):
    meanStdDict[0] = []
    meanStdDict[1] = []
    # empty lists
    m0 = []
    m1 = []
    classes_count = np.unique(data_matrix[:, len(data_matrix[0]) - 1]).size
    xi = 0
    while xi < len(im):
        if im[xi][len(im[0]) - 1] == '0':
            m0.append(im[xi])
        elif im[xi][len(im[0]) - 1] == '1':
            m1.append(im[xi])
        xi = xi + 1
    m0 = np.array(m0)  # labels 0
    m1 = np.array(m1)  # labels 1
    main_arr_length = len(main_arr)
    x = 0
    while x < (main_arr_length - 1):
        y = 0
        while y < classes_count:
            if main_arr[x] == "Numeral":
                t = []
                if y == 0:
                    mc = m0[:, x]
                    mc = mc.astype(np.float)
                    std = np.std(mc, ddof=1)
                    mean = np.mean(mc)
                    t.append(mean)
                    t.append(std)
                    meanStdDict[y].append(t)
                else:
                    mc = m1[:, x]
                    mc = mc.astype(np.float)
                    std = np.std(mc, ddof=1)
                    mean = np.mean(mc)
                    t.append(mean)
                    t.append(std)
                    meanStdDict[y].append(t)
            elif main_arr[x] == "Categorical":
                meanStdDict[y].append(["Categorical"])
            y = y + 1
        x = x + 1

# method to calculate prior probability
def prior(class_label):
    numerator = 0
    denominator = 0
    col = data_matrix[:, len(data_matrix[0]) - 1]
    le = list(col)
    denominator = len(col)
    numerator = le.count(class_label)
    return numerator / denominator

def priorc(q, im):
    res = 1.0
    x = 0
    while x < len(q):
        column = im[:, x]
        le = list(column)
        numerator = le.count(q[x])
        denominator = len(le)
        res *= (numerator/denominator)
        x = x + 1
    return res

def posterior(testData, trainData):
    ml = []
    for datum in testData:
        q = list(datum)
        q.pop()
        classes_count = np.unique(data_matrix[:, len(data_matrix[0]) - 1]).size
        final = []
        x = 0
        while x < classes_count:
            probability = 1.0
            y = 0
            while y < len(q):
                if main_arr[y] == "Numeral":
                    msl = meanStdDict.get(x)[y]
                    mu_value = msl[0]
                    sigma_value = msl[1]
                    xx = q[y]
                    probability *= ss.norm(mu_value, sigma_value).pdf(float(xx))
                else:
                    probability *= categorical(q[y], y, str(x))
                y = y + 1
            prior_probability = prior(str(x))
            final.append(prior_probability * probability)
            x = x + 1
        ml.append(final.index(np.amax(final)))
    return ml

mviddict(data_matrix)
q = input("Please enter query: ")
q = q.split(',')
if len(q) == len(data_matrix[0]) - 1:
    classes_count = np.unique(data_matrix[:, len(data_matrix[0]) - 1]).size
    final = []
    ml = []
    x = 0
    while x < classes_count:
        probability = 1.0
        for y in range(len(q)):
            if main_arr[y] == "Numeral":
                msl = meanStdDict.get(x)[y]
                mu_value = msl[0]
                sigma_value = msl[1]
                xx = q[y]
                probability *= ss.norm(mu_value, sigma_value).pdf(float(xx))
            else:
                m = categorical(q[y], y, str(x))
                probability *= m
        prior_p = prior(str(x))
        dp = priorc(q, data_matrix)
        final.append((prior_p * probability)/dp)
        x = x + 1
    print("Class label 0 Probability is : ", final[0] / (final[0] + final[1]))
    print("Class label 1 Probability is : ", final[1] / (final[0] + final[1]))
    ml.append(final.index(np.amax(final)))
    print("Final Class is : ", ml[0])

