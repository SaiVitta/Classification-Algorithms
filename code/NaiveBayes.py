import numpy as np
from sklearn.cross_validation import train_test_split
import scipy as sc
from sklearn import preprocessing
from sklearn.cross_validation import KFold
import sys

def naiveBayes(train_samples, train_labels, test_samples, test_labels):

    predicted_labels = []
    records_0 = []
    records_1 = []

    for i in range(train_samples.shape[0]):
        if(int(train_labels[i][0]) == 1):
            records_1.append(train_samples[i])
        else:
            records_0.append(train_samples[i])

    mean_records_0 = np.array(records_0).mean(axis = 0)
    std_dev_records_0 = np.array(records_0).std(axis = 0)
    mean_records_1 = np.array(records_1).mean(axis = 0)
    std_dev_records_1 = np.array(records_1).std(axis = 0)    

    for g in range(test_samples.shape[0]):
        probability_class_0 = 1.0
        probability_class_1 = 1.0

        for i in range(test_samples.shape[1]):
            probability_class_0 *= sc.stats.norm(mean_records_0[i], std_dev_records_0[i]).pdf(test_samples[g, i])

        probability_class_0 *= len(records_0) / float(train_samples.shape[0])

        for i in range(test_samples.shape[1]):
            probability_class_1 *= sc.stats.norm(mean_records_1[i], std_dev_records_1[i]).pdf(test_samples[g, i])

        probability_class_1 *= len(records_1) / float(train_samples.shape[0])

        if probability_class_1 > probability_class_0:
            predicted_labels.append(1)
        elif probability_class_0 > probability_class_1:
            predicted_labels.append(0)

    recall_original_positive_count = 0
    recall_true_positive_count = 0
    true_count = 0
    precision_positive_count = 0
    precision_true_positive_count = 0
    
    for i in range(len(predicted_labels)):
        
        if test_labels[i] == 1 and (predicted_labels[i] == 0 or predicted_labels[i] == 1):
            recall_original_positive_count += 1
            if predicted_labels[i] == 1:
                recall_true_positive_count += 1
                
        if (predicted_labels[i] == 1 and test_labels[i] == 1) or (predicted_labels[i] == 0 and test_labels[i] == 0):
            true_count += 1

        if predicted_labels[i] == 1.0 and (test_labels[i] == 0 or test_labels[i] == 1):
            precision_positive_count += 1
            if test_labels[i] == 1:
                precision_true_positive_count += 1

    total_count = len(predicted_labels)
    accuracy = float(true_count) / total_count
    precision = float(precision_true_positive_count) / precision_positive_count
    recall = float(recall_true_positive_count) / recall_original_positive_count
    f1 = 2*((precision*recall) / (precision+recall))

    return accuracy,precision,recall,f1

if __name__ == '__main__':
    file_name = sys.argv[1]
    dataset_file = open(file_name, 'r')
    given_data = [line.split('\t') for line in dataset_file.readlines()]
    transormed_data = [list(x) for x in zip(*given_data)]

    for row in transormed_data:
        indx = transormed_data.index(row)
        for val in row:
             if any(char.isalpha() for char in val) == True:
                 le = preprocessing.LabelEncoder()
                 le.fit(list(set(row)))
                 temporary =  le.transform(row).tolist()
                 for i in range(0, len(temporary)):
                     temporary[i] = int(temporary[i]) + 1
                 transormed_data[indx] = temporary
    given_data = [list(x) for x in zip(*transormed_data)]

    for sample in given_data:
        sample[-1] = sample[-1][:-1]
        for i in range(0,len(sample)):
            sample[i] = float(sample[i])

    data = np.array(given_data)
    samples_total = data[:,:-1]
    labels_total = data[:,-1:]
    training_samples, test_samples, training_labels, test_labels = train_test_split(samples_total, labels_total, test_size=0.30, random_state=0)

    train_samples = np.array(training_samples)
    train_labels = np.array(training_labels)
    test_samples = np.array(test_samples)
    test_labels = np.array(test_labels)

    num_samples = samples_total.shape[0]
    nFolds = 10
    k_fold = KFold(num_samples, n_folds = nFolds)
    iteration = 1

    accuracy_total = 0
    precision_total = 0 
    recall_total = 0 
    f1_total = 0

    for train_index, test_index in k_fold:
        train_samples, test_samples = samples_total[train_index], samples_total[test_index]
        train_labels, test_labels = labels_total[train_index], labels_total[test_index]

        accuracy, precision, recall, f1 = naiveBayes(train_samples, train_labels, test_samples, test_labels)
        iteration += 1
        accuracy_total += accuracy
        precision_total += precision
        recall_total += recall
        f1_total += f1
        avg_accuracy = accuracy_total / nFolds
        avg_precision = precision_total / nFolds
        avg_recall = recall_total / nFolds
        avg_f1 = f1_total / nFolds

    print("Final Performance Parameters:")
    print ("Average accuracy: " , avg_accuracy)
    print ("Average precision : " , avg_precision)
    print ("Average recall: " , avg_recall)
    print ("Average F1 measure: " , avg_f1)