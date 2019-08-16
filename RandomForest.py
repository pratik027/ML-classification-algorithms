import DataLoad
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


def run_random_forest():
    train, val, test, usps = DataLoad.create_dataset()
    classifier = RandomForestClassifier(n_estimators=10)
    classifier.fit(train[0], train[1])

    y_pred = classifier.predict(test[0])
    print("accuracy test ", DataLoad.get_accuracy(y_pred, test[1]))
    print(confusion_matrix(test[1], y_pred))
    DataLoad.write_to_csv("random_forest_test.csv", y_pred)

    # -------------------------
    y_pred = classifier.predict(usps[0])
    print("accuracy usps ", DataLoad.get_accuracy(y_pred, usps[1]))
    print(confusion_matrix(usps[1], y_pred))
    DataLoad.write_to_csv("random_forest_test_usps.csv", y_pred)


def main():
    run_random_forest()


if __name__ == '__main__':
    main()
