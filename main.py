import json
from dnn.dnn import NeuralNet
from dnn.utils.func_utils import sigmoid, classification_cost_function
from data_access import get_data_medical, get_json_data


def get_best_model(no_of_runs):

    data = get_data_medical()
    input_data = data[0]
    output_data = data[1]
    output_for_train = data[2]

    theta_border = 0.12
    epoch_count = 1000
    k = 10

    nn_data = get_json_data("./resources/network_data.json")

    top_score = -1
    top_thetas = []

    for i in range(0, no_of_runs):

        net = NeuralNet(input_data.shape[1], output_for_train.shape[1],
                        nn_data["hidden_layers"], theta_border, nn_data["lambda"],
                        activation_function=sigmoid,
                        cost_function=classification_cost_function)

        accuracy = net.k_fold_cv_train(input_data, output_for_train, epoch_count, k)

        if accuracy > top_score:
            top_score = accuracy
            top_thetas = net.get_weights()

    return [top_score, top_thetas]


if __name__ == "__main__":

    run_count = 10
    top = get_best_model(run_count)

    score = top[0]
    thetas = top[1]

    print("\nBest score:")
    print(score)

    with open("./resources/model_output.json", "w") as data_file:
        json.dump({"score": score, "thetas": [x.tolist() for x in thetas]}, data_file)
