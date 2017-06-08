import numpy as np


def mean_normalization(data):
    """
    Performs mean normalization on each column of the data matrix
    :param data: The data matrix
    :return: The mean-normalized data matrix
    """

    means = np.zeros((1, data.shape[1]))
    stds = np.zeros((1, data.shape[1]))

    cpy_data = np.zeros(data.shape)

    for i in range(0, data.shape[1]):
        means[0, i] = np.mean(data[:, i])
        stds[0, i] = np.std(data[:, i])

    for i in range(0, data.shape[0]):
        for j in range(0, data.shape[1]):
            cpy_data[i, j] = (data[i, j] - means[0, j]) / stds[0, j]

    return cpy_data
