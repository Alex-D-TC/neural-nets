import numpy as np
from dnn.utils.func_utils import sigmoid, classification_cost_function, get_gradient_func
from dnn.utils.matrix_utils import rand_init_matrix, roll_matrix_list, unroll_matrix_list


def init_neural_net(input_neuron_count, output_neuron_count, hidden_layers, theta_border):
    """
    Constructs the theta matrices representing a given neural network architecture
    :param input_neuron_count: The input neuron count
    :param output_neuron_count: The output neuron count
    :param hidden_layers: A list of the sizes of the hidden layers of the network
    :param theta_border: The range of values that theta should have. The interval is [-theta_border, theta_border]
    :return: A list of the theta matrices
    """

    hidden_layer_count = len(hidden_layers)

    # First theta
    theta = [rand_init_matrix((input_neuron_count + 1, hidden_layers[0]), theta_border)]

    # Intermediary thetas
    for i in range(0, hidden_layer_count - 1):
        theta.append(rand_init_matrix((hidden_layers[i] + 1, hidden_layers[i + 1]), theta_border))

    # Last Theta
    theta.append(rand_init_matrix((hidden_layers[-1] + 1, output_neuron_count), theta_border))

    return theta


def feedforward(theta, input_data, activation=sigmoid):
    """
    Performs the feedforward operation of a neural network
    All inputs must be numpy arrays unless otherwise specified
    :param theta: The theta variables of the neural network. An array of 2D numpy matrices
    :param input: The input data
    :param activation: The activation function. Default is the sigmoid function
    :return: A matrix of results, one line for the result of each line in the input
    """

    k = len(theta)
    result = input_data
    for i in range(0, k):
        result = np.insert(result, 0, 1, axis=1)
        result = activation(np.dot(result, theta[i]))

    # Process final activation for classification purposes
    for i in range(0, result.shape[0]):
        max_value = np.max(result[i, :])
        set_one = False
        for j in range(0, result.shape[1]):
            if not set_one and result[i, j] == max_value:
                result[i, j] = 1
                set_one = True
            else:
                result[i, j] = 0

    return result


def backprop(input_data, theta, output, lambd, matrix_shapes, activation=sigmoid, cost_function=classification_cost_function):
    """
    Performs the backpropagation algorithm given a dataset and the network's weights
    :param input_data: The input data matrix
    :param theta: The network weights, as a 1D vector
    :param output: The labeled output for each input data row
    :param lambd: The regularization parameter
    :param matrix_shapes: The shapes of the theta matrices, so they can be unrolled for easier computation
    :param activation: The activation function of the neurons
    :param cost_function: The cost function of the neural network
    :return: The result of the cost function and its gradient as a 1D vector of the same size as theta
    """

    theta = unroll_matrix_list(theta, matrix_shapes)
    gradient_func = get_gradient_func(activation)

    # Compute feedforward
    k = len(theta)
    result = np.insert(input_data, 0, 1, axis=1)
    a_vector = [result]
    z_vector = [sigmoid(result)]
    for i in range(0, k):
        result = np.dot(result, theta[i])
        if i != k - 1:
            # Add the bias node activation
            result = np.insert(result, 0, 1, axis=1)
        z_vector.append(result)
        result = activation(result)
        a_vector.append(result)

    # Compute cost
    cost = cost_function(output, a_vector[-1], lambd, theta)

    # Compute deltas with backpropagation
    theta_count = len(theta)
    delta = [a_vector[-1] - output]
    for i in range(theta_count - 1, 0, -1):
        sgm_grad = np.array(list(map(lambda x: gradient_func(x), z_vector[i])))
        dlt = delta[-1]
        if i != theta_count - 1:
            dlt = dlt[:, 1:]
        errors = np.dot(dlt, np.transpose(theta[i]))
        delta.append(np.multiply(errors, sgm_grad))

    # With the deltas, compute the gradient
    delta.reverse()

    gradients = []
    for i in range(0, theta_count):
        # Cut the first column. It's of the bias error
        if i != theta_count - 1:
            delt = delta[i][:, 1:]
        else:
            delt = delta[i]
        gradients.append(np.dot(np.transpose(a_vector[i]), delt))

    # Regularize the gradient
    input_data_count = input_data.shape[0]
    gradients = list(map(lambda x: x / input_data_count, gradients))
    for i in range(1, len(gradients)):
        gradients[i] = gradients[i] + lambd * theta[i]

    return [cost, roll_matrix_list(gradients)]
