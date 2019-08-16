import tensorflow as tf
import pandas as pd
# from tqdm import tqdm_notebook
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import DataLoad

TrainingPercent = 70
ValidationPercent = 20
TestPercent = 10


class NeuralNetwork:
    """
    This class performs Neural Network on given data-set
    """
    def __init__(self):
        self.input_hidden_weights = None
        self.hidden_output_weights = None
        self.hidden_layer = None
        self.output_layer = None
        self.error_function = None
        self.training = None
        self.prediction = None

    def init_weights(self, shape):
        """
        Initialize weights
        :param shape: of expected weights
        :return: weights
        """
        return tf.Variable(tf.random_normal(shape, stddev=0.01))

    def define_model(self, num_features=9, num_buckets=2, learning_rate=0.05):
        """
        Define Neural Networks
        :param num_features: number of features inputes
        :param num_buckets: number of buckets for output
        :param learning_rate: learning rate
        :return:
        """
        NUM_HIDDEN_NEURONS_LAYER_1 = 400

        self.inputTensor = tf.placeholder(tf.float32, [None, num_features])
        self.outputTensor = tf.placeholder(tf.float32, [None, num_buckets])
        # Initializing the input to hidden layer weights
        self.input_hidden_weights = self.init_weights([num_features, NUM_HIDDEN_NEURONS_LAYER_1])
        self.input_hidden_weights_2 = self.init_weights([NUM_HIDDEN_NEURONS_LAYER_1, NUM_HIDDEN_NEURONS_LAYER_1])
        # Initializing the hidden to output layer weights
        self.hidden_output_weights = self.init_weights([NUM_HIDDEN_NEURONS_LAYER_1, num_buckets])

        # Computing values at the hidden layer
        self.hidden_layer = tf.nn.relu(tf.matmul(self.inputTensor, self.input_hidden_weights))
        #self.hidden_layer_2 = tf.nn.relu(tf.matmul(self.hidden_layer, self.input_hidden_weights_2))
        # Computing values at the output layer

        self.output_layer = tf.matmul(self.hidden_layer, self.hidden_output_weights)

        # Defining Error Function
        self.error_function = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output_layer, labels=self.outputTensor))

        # Defining Learning Algorithm and Training Parameters
        self.training = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.error_function)

        # Prediction Function
        self.prediction = tf.argmax(self.output_layer, 1)

    def train_model(self,training_data,training_label,testing_data,usps, NUM_EPOC=5000, BATCH_SIZE=128):
        """
        Train Neural network model
        :param training_data: features for training
        :param training_label: training target
        :param testing_data: testing dataset
        :param NUM_EPOC: Number of epoch
        :param BATCH_SIZE: size of batch
        :return: predicted label, training accuracy
        """
        training_accuracy = []
        with tf.Session() as sess:
            # Set Global Variables ?
            tf.global_variables_initializer().run()
            # '------------training started-------------')
            for epoch in range(NUM_EPOC):

                # Shuffle the Training Dataset at each epoch
                p = np.random.permutation(range(len(training_data)))
                training_data = training_data[p]
                training_label = training_label[p]
                # Start batch training
                for start in range(0, len(training_data), BATCH_SIZE):

                    end = start + BATCH_SIZE
                    sess.run(self.training, feed_dict={self.inputTensor: training_data[start:end],
                                                       self.outputTensor: training_label[start:end]})
                # append training accuracy for current epoch
                training_accuracy.append(np.mean(np.argmax(training_label, axis=1) ==
                                                 sess.run(self.prediction, feed_dict={self.inputTensor: training_data,
                                                                                 self.outputTensor: training_label})))
            # Testing
            predicted_test_label = sess.run(self.prediction, feed_dict={self.inputTensor: testing_data})
            DataLoad.write_to_csv("nn_test.csv",predicted_test_label)
            predicted_usps_label = sess.run(self.prediction, feed_dict={self.inputTensor: usps[0]})
            DataLoad.write_to_csv("nn_test_usps.csv", predicted_usps_label)
        return predicted_test_label, training_accuracy, predicted_usps_label

    def start_neural_network(self, learning_rate=0.002, num_epoch=50,show_graph=False):
        """
        Load features (concat/subtract) and target from dataset (HBD or GSC)
        Start training Neural Network model
        Print accuracy on test dataset
        :param dataset: HBD or GSC dataset
        :param op: concat or subtract feature
        :param limit: size of dataset for training and testing
        :param learning_rate: learning rate
        :param num_buckets: number of output bucket
        :param show_graph: boolean to display accuracy graph
        :return:
        """
        print("Neural Network ")
        train, validation, test, usps = DataLoad.create_dataset()
        train_target = DataLoad.convert_target(train[1])
        test_target = DataLoad.convert_target(test[1])
        print("start define model")
        self.define_model(num_features=train[0].shape[1], num_buckets=train_target.shape[1], learning_rate=learning_rate)
        print("start training model")
        predicted_test_label, training_accuracy, usps_test_label = self.train_model(training_data=train[0], training_label=train_target,
                       testing_data=test[0],usps=usps, NUM_EPOC=num_epoch, BATCH_SIZE=100)

        #xpred=DataLoad.pick_max_result(predicted_test_label)
        print("Testing Accuracy: " , DataLoad.get_accuracy(predicted_test_label, test[1]))
        print(confusion_matrix(test[1], predicted_test_label))

        #xpred = DataLoad.pick_max_result(usps_test_label)
        print("Testing USPS Accuracy: ", DataLoad.get_accuracy(usps_test_label, usps[1]))
        print(confusion_matrix(usps[1], usps_test_label))

        if show_graph:
            import matplotlib.pyplot as plt
            plt.plot(training_accuracy)
            plt.show()


def main():
    nn = NeuralNetwork()
    nn.start_neural_network(num_epoch=100, learning_rate=0.015,show_graph=False)


if __name__ == '__main__':
    main()
