# Sirohi, Krishnakant Singh
# 1001-668-969
# 2019-09-22
# Assignment-01-01

import numpy as np
import itertools


class Perceptron(object):
    def __init__(self, input_dimensions=2, number_of_classes=4, seed=None):
        """
        Initialize Perceptron model
        :param input_dimensions: The number of features of the input data, for example (height, weight) would be two features.
        :param number_of_classes: The number of classes.
        :param seed: Random number generator seed.
        """
        if seed != None:
            np.random.seed(seed)
        self.input_dimensions = input_dimensions
        self.number_of_classes = number_of_classes
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize the weights, initialize using random numbers.
        Note that number of neurons in the model is equal to the number of classes
        """
        self.weights = np.random.randn(self.number_of_classes, self.input_dimensions + 1)

    def initialize_all_weights_to_zeros(self):
        """
        Initialize the weights, initialize using random numbers.
        """
        self.weights = np.zeros(shape=(self.number_of_classes, self.input_dimensions + 1))

    def predict(self, X):
        """
        Make a prediction on an array of inputs
        :param X: Array of input [input_dimensions,n_samples]. Note that the input X does not include a row of ones
        as the first row.
        :return: Array of model outputs [number_of_classes ,n_samples]
        """
        X_with_1 = np.insert(X, 0, 1, axis=0)
        a = np.where((np.dot(self.weights, X_with_1)) >= 0.0, 1, 0)
        return a

    def print_weights(self):
        """
        This function prints the weight matrix (Bias is included in the weight matrix).
        """
        print(self.weights)
        """raise Warning("You must implement print_weights")"""

    def train(self, X, Y, num_epochs=10, alpha=0.001):
        """
        Given a batch of data, and the necessary hyperparameters,
        this function adjusts the self.weights using Perceptron learning rule.
        Training should be repeated num_epochs time.
        :param X: Array of input [input_dimensions,n_samples]
        :param Y: Array of desired (target) outputs [number_of_classes ,n_samples]
        :param num_epochs: Number of times training should be repeated over all input data
        :param alpha: Learning rate
        :return: None
        """
        X_with_1 = np.insert(X, 0, 1, axis=0)
        for i in range(num_epochs):
            for j in range(X_with_1.shape[1]):
                desired_output_for_sample = np.array(Y[:, j]).reshape(self.number_of_classes, 1)
                error_for_sample = np.array(desired_output_for_sample - self.predict(np.transpose(X)[j]).reshape(self.number_of_classes, 1)).reshape(self.number_of_classes, 1)
                u = error_for_sample * np.transpose(X_with_1)[j] * alpha
                self.weights = self.weights + u

    def calculate_percent_error(self, X, Y):
        """
        Given a batch of data this function calculates percent error.
        For each input sample, if the output is not the same as the desired output, Y,
        then it is considered one error. Percent error is number_of_errors/ number_of_samples.
        :param X: Array of input [input_dimensions,n_samples]
        :param Y: Array of desired (target) outputs [number_of_classes ,n_samples]
        :return percent_error
        """
        e = np.array(self.predict(X))
        e = np.transpose(Y - e)
        n = 0
        for i in e:
            if any(i) != 0:
                n += 1
        return n / X.shape[1]


if __name__ == "__main__":
    """
    This main program is a sample of how to run your program.
    You may modify this main program as you desire.
    """

    input_dimensions = 2
    number_of_classes = 2

    model = Perceptron(input_dimensions=input_dimensions, number_of_classes=number_of_classes, seed=1)
    X_train = np.array([[0,0], [1,1],[0,1.33],[1,0.67]]).T
    """X_train = np.array([[1, 1, 2, 2, -1, -2, -1, -2],
                        [1, 2, -2, 0, 2, 1, -1, -2]])"""
    Y_train = np.array([[0,0],[0,0],[0,1],[0,1]])
    """Y_train = np.array([[0, 0, 0, 0, 1, 1, 1, 1],
                        [0, 0, 1, 1, 0, 0, 1, 1]])"""
    model.initialize_all_weights_to_zeros()
    print("****** Model weights ******\n", model.weights)
    print("****** Input samples ******\n", X_train)
    print("****** Desired Output ******\n", Y_train)
    percent_error = []
    for k in range(1):
        model.train(X_train, Y_train, num_epochs=1, alpha=0.0001)
        percent_error.append(model.calculate_percent_error(X_train, Y_train))
    print("******  Percent Error ******\n", percent_error)
    print("****** Model weights ******\n", model.weights)