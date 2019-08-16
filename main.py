import LogisticRegressionMulticlass
import RandomForest
import svm
import NeuralNetwork
import combine_model


def main():
    model = LogisticRegressionMulticlass.LogisticRegression()
    model.start_logistic_regression(learning_rate=0.7, num_epoch=50, theta=1, show_graph=False)

    svm.run_svm(kernal=svm.linear, train_size=20000)
    RandomForest.run_random_forest()

    nn = NeuralNetwork.NeuralNetwork()
    nn.start_neural_network(num_epoch=100, learning_rate=0.015, show_graph=False)

    combine_model.combine_model(combine_model.MNIST)
    combine_model.combine_model(combine_model.USPS)


if __name__ == '__main__':
    main()
