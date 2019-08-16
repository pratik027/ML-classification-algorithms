from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import DataLoad

TrainingPercent = 80
ValidationPercent = 10
TestPercent = 10


class LogisticRegression:
    """
    This class performs Logistic regression on given data-set
    """
    def __init__(self):
        self.weights = None

    def softmax(self, data, theta=0.015):
        z_new = []
        for y in data:
            exp_y = np.exp(y)
            exp_sum = np.sum(exp_y)
            z_new.append(exp_y/exp_sum)
        return z_new


    def sigmoid(self, z):
        """
        Calculate sigmoid of values in
        :param z: vector of (feature * weights)
        :return: sigmoid of all values in vector z
        """
        return 1 / (1 + np.exp(-z))

    def cross_entropy_loss(self, h, target):
        """
        Calculate loss using cross entropy
        :param h: hypothesis
        :param target: target values
        :return: vector of loss values
        """
        return (-target * np.log(h) - (1 - target) * np.log(1 - h)).mean()

    def train_model(self, feature_d, target, learning_rate, num_epoch=400,theta=0.1):
        """
        Train logistic regression model
        :param feature_d: features
        :param target: target values for given feature
        :param learning_rate: learning rate for model
        :param num_epoch: number of epoch
        :return: list of accuracy at each epoch
        """
        # weights initialization
        self.weights = np.zeros((feature_d.shape[1], target.shape[1]))
        training_accuracy = []

        for i in range(num_epoch):
            z = np.dot(feature_d, self.weights)
            #hypothesis = self.sigmoid(z)
            hypothesis = self.softmax(z,theta=theta)
            gradient = np.dot(feature_d.T, (hypothesis - target)) / target.shape[0]
            self.weights -= learning_rate * gradient
            #print("hypothesis shape", hypothesis.shape, "target shape", target.shape, "weights ", self.weights.shape,
            #      "grad shape", gradient.shape,"new weights",self.weights.shape)
            training_accuracy.append(DataLoad.get_accuracy_logistic(np.round(hypothesis), target))
        return training_accuracy

    def get_prediction(self, feature_mat):
        """
        calculate prediction for given dataset using current weights
        :param feature_mat: feature matrix
        :return: prediction for given feature matrix
        """
        return np.round(self.sigmoid(np.dot(feature_mat, self.weights)))

    def get_prediction_softmax(self, feature_mat, theta=0.2):
        """
        calculate prediction for given dataset using current weights
        :param feature_mat: feature matrix
        :return: prediction for given feature matrix
        """
        return np.round(self.softmax(np.dot(feature_mat, self.weights),theta))

    def start_logistic_regression(self, learning_rate=0.005, num_epoch=700, theta=0.1, show_graph=False):
        """
        Load features (concat/subtract) and target from dataset (HBD or GSC)
        Start training logistic regression model
        Print accuracy on test dataset
        :param learning_rate: learning rate for model training
        :param num_epoch: number of epoch
        :param show_graph: boolean to display graph of training
        :return:
        """
        # Prepare data sets ---------------
        train, validation, test, USPS = DataLoad.create_dataset()
        train_t = DataLoad.convert_target(train[1])
        # train model --------------------------
        print("Logistic Regression started ")
        training_accuracy = self.train_model(train[0], train_t, learning_rate, num_epoch, theta)

        # ------- prediction for amnist test data
        predictions = self.get_prediction_softmax(test[0], theta)
        test_t = DataLoad.convert_target(test[1])
        acc = DataLoad.get_accuracy_logistic(test_t, predictions)
        print('accuracy', acc)

        if show_graph:
            plt.plot(training_accuracy)
            plt.show()

        xpred_test = DataLoad.pick_max_result(predictions)
        DataLoad.write_to_csv("logistic_test.csv", xpred_test)

        conf_mat = metrics.confusion_matrix(test[1], xpred_test)
        print(conf_mat)

        # --------- test usps data
        predictions = self.get_prediction_softmax(USPS[0], theta)
        usps_t = DataLoad.convert_target(USPS[1])
        acc = DataLoad.get_accuracy_logistic(usps_t, predictions)
        print("usps accuracy:", acc)
        xpred_usps = DataLoad.pick_max_result(predictions)
        DataLoad.write_to_csv("logistic_test_usps.csv", xpred_usps)

        conf_mat = metrics.confusion_matrix(USPS[1], xpred_usps)
        print(conf_mat)


def main():
    model = LogisticRegression()
    model.start_logistic_regression(learning_rate=0.7, num_epoch=50, theta=1, show_graph=False)


if __name__ == '__main__':
    main()
