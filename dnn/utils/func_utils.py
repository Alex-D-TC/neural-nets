import numpy as np


def sigmoid_gradient(x):
    """
    The sigmoid gradient function
    :param x: The input data
    :return: sigmoid'(x)
    """
    sigm = sigmoid(x)
    return sigm - sigm * sigm


def sigmoid(x):
    """
    The sigmoid function
    :param x: The input data
    :return: 1 / (1 + e^(-x))
    """
    return 1.0 / (1.0 + np.exp(-x))


def classification_cost_function(output, output_activation, lambd, theta):
    """
    A classic classification cost function
    :param output: The labeled output data
    :param output_activation: The output activations
    :param lambd: The regularization parameter
    :param theta: The network weights
    :return: The cost
    """

    output_activation = np.where(output_activation < 1, output_activation, output_activation - 0.0000001)

    log_activation = np.log(output_activation)
    log_activation_minus = np.log(1.0 - output_activation)
    size = output.shape[0]

    cost = sum(map(lambda y, log, log_minus:
                   (-1 * np.dot(y, np.transpose(log))) - np.dot((1 - y), np.transpose(log_minus)),
                   output, log_activation, log_activation_minus))

    # Regularization
    return cost + (lambd / (2 * size)) * sum(list(map(lambda t: np.sum(np.square(t)), theta)))


def get_gradient_func(func):

    if func == sigmoid:
        return sigmoid_gradient
