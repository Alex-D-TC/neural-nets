import numpy as np
from dnn.utils.stat_utils import mean_normalization
import json
import scipy.io


def get_json_data(path):

    with open(path) as d_file:
        json_data = json.load(d_file)

    return json_data


def get_data_medical():

    with open("./resources/column_3C.dat") as data_file:
        data = data_file.readlines()

    class_mapping = get_json_data("./resources/class_mapping.json")

    input_data = np.zeros((len(data), len(data[0].split('\n')[0].split(' ')) - 1))
    output_data = []

    for i in range(0, len(data)):
        line_split = data[i].split('\n')[0].split(' ')

        output_data.append(line_split[-1])

        for j in range(0, len(line_split) - 1):
            input_data[i, j] = float(line_split[j])

    input_data = mean_normalization(input_data)

    output_data = np.array(list(map(lambda x: class_mapping[x], output_data)))

    output_for_train = translate_classes(output_data)

    return [input_data, output_data, output_for_train]


def get_mat_data(path):

    mat_data = scipy.io.loadmat(path)
    input_data = mat_data["X"]
    output_data = mat_data["y"]

    return [input_data, output_data]


def translate_classes(output_data):

    output_for_train = np.zeros((output_data.shape[0], np.max(output_data)))
    for i in range(0, output_data.shape[0]):
        index = output_data[i] - 1
        output_for_train[i][index] = 1

    return output_for_train


def get_data_images():

    mat = scipy.io.loadmat("./resources/ex4data1.mat")

    input_data = mat["X"]
    output_data = mat["y"]

    output_for_train = translate_classes(output_data)

    return [input_data, output_data, output_for_train]


