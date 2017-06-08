import json
from dnn.utils.net_operations import feedforward
from data_access import get_data_medical
import numpy as np

if __name__ == "__main__":

    with open("./resources/model_output.json") as data_file:
        data = json.load(data_file)

    net_weights = data["thetas"]

    data = get_data_medical()

    input_data = data[0]
    output_data = data[1]
    output_for_train = data[2]

    result = feedforward(net_weights, input_data)

    count = 0
    for i in range(0, result.shape[0]):
        if np.array_equal(output_for_train[i, :], result[i, :]):
            count = count + 1

    print("Model effectiveness:\n")
    print(count / result.shape[0])

