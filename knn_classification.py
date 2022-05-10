import csv
import random

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets, metrics
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

classes = {}
data_pair = []
with open('features_organized.tsv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        features = []
        for i in range(1, 13):
            features.append(int(float(row[i])))
        if row[13] not in classes.keys():
            classes[row[13]] = len(classes.keys()) + 1
        data_pair.append((features, classes[row[13]]))

print(classes)


def kNN(x, x_train, y_train, k=5):
    y_pred = np.zeros(len(x), dtype=np.int8)
    for i, sample in enumerate(x):
        dist = [np.linalg.norm(sample - train) for train in x_train]
        k_nearest_labels = []
        for j in range(k):
            closest = np.argmin(dist)
            k_nearest_labels.append(y_train[closest])
            dist.pop(closest)
        labels, counts = np.unique(k_nearest_labels, return_counts=True)
        y_pred[i] = labels[np.argmax(counts)]
    return y_pred


def prepare_10fold(data_pair):
    eind = 0
    random.shuffle(data_pair)
    fold_size = int(len(data_pair) / 10)
    for fold in range(0, 10):
        sind = eind
        eind = sind + fold_size
        train_pair = data_pair[0:sind] + data_pair[eind:len(data_pair)]
        test_pair = data_pair[sind:eind]
        yield (train_pair, test_pair)


data_10fold = prepare_10fold(data_pair)
precision = []
recall = []
f1 = []
for item in data_10fold:
    X = []
    y = []
    for image in item[0]:
        X.append(image[0])
        y.append(image[1])

    X = np.array(X)
    y = np.array(y)

    X_test = []
    y_test = []
    for image in item[1]:
        X_test.append(image[0])
        y_test.append(image[1])

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    predicted = kNN(X_test, X, y, k=1)
    # print(predicted)
    print(metrics.classification_report(y_test, predicted))
    print("Precision:", round(metrics.precision_score(y_test, predicted, average="micro"), 3),
          "Recall:", round(metrics.recall_score(y_test, predicted, average="micro"), 3),
          "F1-score:", round(metrics.f1_score(y_test, predicted, average="micro"), 3))

    precision.append(round(metrics.precision_score(y_test, predicted, average="micro"), 3))
    recall.append(round(metrics.recall_score(y_test, predicted, average="micro"), 3))
    f1.append(round(metrics.f1_score(y_test, predicted, average="micro"), 3))

print("Precision:", np.average(np.array(precision)), "Recall:", np.average(np.array(recall)), "F1-score:",
      np.average(np.array(f1)))
