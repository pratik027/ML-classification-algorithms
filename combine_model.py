from sklearn.metrics import classification_report, confusion_matrix
import DataLoad
from statistics import mode
import seaborn as sn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



MNIST = ['logistic_test.csv','random_forest_test.csv','svm_test.csv','nn_test.csv']
USPS = ['logistic_test_usps.csv','random_forest_test_usps.csv','svm_test_usps.csv','nn_test_usps.csv']


def load_data(file):
    l_ = []
    with open(file,'r') as f:
        for line in f.readlines():
            try:
                l_.append(int(line.strip()))
            except Exception as e:
                print(e)
    return l_


def get_mode(data):
    try:
        return mode(data)
    except Exception as e:
        # return NN output, if there is no mode
        #print(e)
        return data[-1]


def combine_model(data_=MNIST):
    _, _, test, usps = DataLoad.create_dataset()
    logistic = load_data(data_[0])
    ran_for = load_data(data_[1])
    svm = load_data(data_[2])
    nn = load_data(data_[3])

    combine_result = []

    for i in range(len(logistic)):
        combine_result.append(get_mode([logistic[i], ran_for[i], svm[i], nn[i]]))

    if data_ == MNIST:
        cm=confusion_matrix(test[1], combine_result)
        print(DataLoad.get_accuracy(test[1],combine_result))
    else:
        cm=(confusion_matrix(usps[1], combine_result))
        print(DataLoad.get_accuracy(usps[1], combine_result))
    print(cm)


if __name__ == "__main__":
    combine_model(USPS)
