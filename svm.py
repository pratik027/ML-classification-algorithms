import DataLoad
import datetime
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

sigmoid = 'sigmoid'
rbf = 'rbf'
linear = 'linear'
k = rbf


def run_svm(kernal=k,train_size=50000):
    train, valid, test, usps = DataLoad.create_dataset()
    svclassifier = SVC(kernel=kernal, gamma=1)
    svclassifier.fit(train[0][:train_size], train[1][:train_size])

    y_pred = svclassifier.predict(test[0])
    DataLoad.write_to_csv("svm_test.csv", y_pred)
    print("accuracy test ", DataLoad.get_accuracy(y_pred, test[1]))
    print(confusion_matrix(test[1], y_pred))

    # -------------------------
    y_pred = svclassifier.predict(usps[0])
    DataLoad.write_to_csv("svm_test_usps.csv", y_pred)
    print("accuracy usps test ", DataLoad.get_accuracy(y_pred, usps[1]))
    print(confusion_matrix(usps[1], y_pred))


def main():
    run_svm(kernal=rbf, train_size=10000)


if __name__ == '__main__':
    main()
