import matplotlib.pyplot as plt
from data_access import get_json_data, get_data_medical
from dnn.utils.net_operations import feedforward
import numpy as np
import json
import sys

if __name__ == "__main__":

    try:
        is_debug = sys.argv[1]
        isDebug = bool(is_debug)
    except ValueError:
        isDebug = False
    except IndexError:
        isDebug = False

    if not isDebug:
        mass_data = get_json_data("./resources/mass_output.json")
    else:
        mass_data = get_json_data("./resources/mass_output_stash.json")

    perf_data = [x["performance"] for x in mass_data]
    lambda_data = [x["lambda"] for x in mass_data]

    fig_1 = plt.figure(1)
    plt.plot(lambda_data, perf_data, 'ro')

    plt.xlabel("Lambda value")
    plt.ylabel("Performance")

    fig_1.show()

    medical_data = get_data_medical()

    input_data = medical_data[0]
    output_for_test = medical_data[2]

    test_performance = []

    for data in mass_data:

        res = feedforward(data["weights"], input_data)
        count = 0

        for i in range(0, input_data.shape[0]):
            if np.array_equal(output_for_test[i, :], res[i, :]):
                count = count + 1

        test_performance.append(count / input_data.shape[0])

    fig_2 = plt.figure(2)

    plt.plot(lambda_data, test_performance, 'ro')
    plt.xlabel("Lamda value")
    plt.ylabel("Performance on whole test data")

    plt.show()

    if not isDebug:

        # Add to stash

        with open("./resources/mass_output_stash.json", "r") as data_file:
            stash = json.load(data_file)

        stash = stash + mass_data
        stash.sort(key=lambda x: x["lambda"])

        with open("./resources/mass_output_stash.json", "w") as data_file:
            json.dump(stash, data_file)
            data_file.flush()
