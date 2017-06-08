import numpy as np
from functools import reduce


def unroll_matrix_list(line, matrix_shapes):
    """
    Transforms a 1D vector in a list of 2D matrices
    :param line: The vector
    :param matrix_shapes: A list of the shapes of the matrix lsit
    :return: The list of matrices
    """
    result_data = []
    i = 0
    curr_shape = 0
    while i < line.shape[0]:
        shape = matrix_shapes[curr_shape]
        curr_shape = curr_shape + 1
        result_data.append(np.reshape(line[i:(i + (shape[0] * shape[1]))], shape))
        i = i + shape[0] * shape[1]
    return result_data


def roll_matrix_list(matrices):
    """
    Transforms a list of 2D matrices into a 1D vector
    :param matrices: The list of matrices
    :return: The 1D vector
    """
    unrolled_list = list(map(lambda x: x.reshape((1, x.shape[0] * x.shape[1])), matrices))
    return reduce(lambda x, y: np.append(x, y), unrolled_list)


def rand_init_matrix(shape, border):
    """
    Randomly initializes a matrix of the given shape, with values from the given interval
    :param shape: The shape of the matrix
    :param border: The border of the values. The interval is [-border, border]
    :return: The matrix
    """
    return np.random.random_sample(shape) * 2 * border - border


def __test_roll_unroll():
    """
    Testing that everything is ok
    """
    data_list = [np.random.randint(0, 100, (3, 3)) for i in range(0, 10)]
    unrolled_data = unroll_matrix_list(roll_matrix_list(data_list), list(map(lambda x: x.shape, data_list)))
    for i in range(0, len(data_list)):
        assert(np.array_equal(data_list[i], unrolled_data[i]))

__test_roll_unroll()
