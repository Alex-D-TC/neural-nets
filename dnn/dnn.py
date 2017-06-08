
from dnn.utils.net_operations import init_neural_net, backprop, feedforward
from dnn.utils.matrix_utils import roll_matrix_list, unroll_matrix_list
from scipy import optimize
import numpy as np


class NeuralNet:

    def __init__(self,
                 input_neuron_count,
                 output_neuron_count,
                 hidden_layers,
                 theta_border,
                 lambd,
                 activation_function,
                 cost_function):

        self.__thetas__ = init_neural_net(
                input_neuron_count, output_neuron_count,
                hidden_layers, theta_border)

        self.__cost_func__ = cost_function
        self.__activation__ = activation_function
        self.__lambda__ = lambd
        self.__theta_shapes__ = list(map(lambda x: x.shape, self.__thetas__))

    def simple_train(self, x_train, y_train, epoch_count):
        """
        Trains a network's parameters in respect to the given input data. 
        Training is done through backpropagation and gradient descent
        :param x_train: The input data
        :param y_train: The labeled outputs
        :param epoch_count: The maximum number of iterations of gradient descent
        """
        rolled_theta = optimize.fmin_tnc(lambda x: backprop(x_train, x, y_train,
                                                            self.__lambda__,
                                                            self.__theta_shapes__,
                                                            cost_function=self.__cost_func__,
                                                            activation=self.__activation__),
                                         roll_matrix_list(self.__thetas__),
                                         maxfun=epoch_count,
                                         messages=0)
        self.__thetas__ = unroll_matrix_list(rolled_theta[0], self.__theta_shapes__)

    def k_fold_cv_train(self, x_train, y_train, epoch_count, k):
        """
        Trains a network's parameters in respect to the given input data, using the k-fold cross-validation method. 
        Training is done through backpropagation and gradient descent
        :param x_train: The input data
        :param y_train: The labeled outputs
        :param epoch_count: The maximum number of iteration of gradient descent
        :param k: The number of batches to split the data into
        """
        concat_data = np.concatenate((x_train, y_train), axis=1)
        np.random.shuffle(concat_data)

        split_data = np.split(concat_data, k)

        input_neuron_count = self.__thetas__[0].shape[0] - 1
        output_neuron_count = self.__thetas__[-1].shape[1]

        k_clusters = []

        for data in split_data:
            k_clusters.append([data[:, :input_neuron_count], data[:, input_neuron_count:]])

        accuracy = np.zeros((0, 1))

        # Do the K training cycles
        for i in range(0, k):

            # Leave the ith cluster for cross-validation
            left_out = k_clusters[i]
            x_train_data = np.zeros((0, input_neuron_count))
            y_train_data = np.zeros((0, output_neuron_count))

            # Build the training batch
            for j in range(0, k):
                if i != j:
                    x_train_data = np.concatenate((x_train_data, k_clusters[j][0]), axis=0)
                    y_train_data = np.concatenate((y_train_data, k_clusters[j][1]), axis=0)

            self.simple_train(x_train_data, y_train_data, epoch_count)

            result = self.eval(left_out[0])
            count = 0
            for j in range(0, result.shape[0]):
                if np.array_equal(result[j, :], left_out[1][j, :]):
                    count = count + 1

            count = count / result.shape[0]
            accuracy = np.concatenate((accuracy, np.array([[count]])), axis=0)

        # Debugging purposes. Print the network accuracy for each batch
        print(accuracy)
        return np.mean(accuracy)

    def eval(self, x):
        """
        Evaluates some input data.
        :param x: The data
        :return: The network's output
        """
        return feedforward(self.__thetas__, x)

    def get_weights(self):
        return self.__thetas__
