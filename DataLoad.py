from PIL import Image
import os
import numpy as np
import pickle
import gzip


def create_dataset():
    filename = 'mnist.pkl.gz'
    f = gzip.open(filename, 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    f.close()
    USPSMat = []
    USPSTar = []
    curPath = 'USPSdata/Numerals'
    savedImg = []

    for j in range(0, 10):
        curFolderPath = curPath + '/' + str(j)
        imgs = os.listdir(curFolderPath)
        for img in imgs:
            curImg = curFolderPath + '/' + img
            if curImg[-3:] == 'png':
                img = Image.open(curImg, 'r')
                img = img.resize((28, 28))
                savedImg = img
                imgdata = (255 - np.array(img.getdata())) / 255
                USPSMat.append(imgdata)
                USPSTar.append(j)
    #print(test_data[0].shape,test_data[1].shape,max(test_data[1]),min(test_data[1]))
    test_feature = np.concatenate((test_data[0], USPSMat))
    test_target = np.concatenate((test_data[1], USPSTar))
    return training_data, validation_data, [np.array(test_data[0]), np.array(test_data[1])], [np.array(USPSMat), np.array(USPSTar)]
    #return training_data, validation_data, [np.array(USPSMat), np.array(USPSTar)]


def convert_target(target):
    new_target =[]
    for t in target:
        target_class = [0]*10
        target_class[t] = 1
        new_target.append(target_class)
    return np.array(new_target)


def get_accuracy(actual_output, target):
    """
    Calculate accuracy based on actual_output, target vectors
    :param actual_output: actual output
    :param target: expected output
    :return: accuracy based on given input
    """
    counter = len([i for i in range(len(actual_output)) if actual_output[i] == target[i]])
    return float(counter * 100) / float(len(target))


def get_accuracy_logistic(actual_output, target):
    """
    Calculate accuracy based on actual_output, target vectors
    :param actual_output: actual output
    :param target: expected output
    :return: accuracy based on given input
    """
    counter = len([i for i in range(len(actual_output)) if np.argmax(actual_output[i]) == np.argmax(target[i])])
    return float(counter * 100) / float(target.shape[0])


def write_to_csv(file_path, data):
    fi = open(file_path,"w")

    for line in data:
        fi.write(str(line)+"\n")
    fi.close()


def pick_max_result(predictions):
    xpred = []
    for p in predictions:
        xpred.append(np.argmax(p))
    return xpred

#print([0]*10)
#create_dataset()