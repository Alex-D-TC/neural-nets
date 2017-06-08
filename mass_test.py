import json
from dnn.dnn import NeuralNet
from dnn.utils.func_utils import sigmoid, classification_cost_function
from data_access import get_data_medical, get_json_data
import threading


def get_best_model(no_of_runs, nn_data):

    data = get_data_medical()
    input_data = data[0]
    output_data = data[1]
    output_for_train = data[2]

    theta_border = 0.12
    epoch_count = 1000
    k = 10

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


writer_lock = threading.Lock()
threads = []


class JsonPrinterThread(threading.Thread):
    def __init__(self, to_write, path):
        threading.Thread.__init__(self)
        self.__data__ = to_write
        self.__path__ = path

    def run(self):
        self._write_data()

    def _write_data(self):
        writer_lock.acquire()
        with open(self.__path__, "a") as data_file:
            json.dump(self.__data__, data_file)
            data_file.write(",\n")
            data_file.flush()
        writer_lock.release()

if __name__ == "__main__":

    _lambda = 0.0100000000000
    lambda_step = 0.01000000000000
    step_count = 10
    nn_data = get_json_data("./resources/network_data.json")
    run_count = 10

    data_list = []

    # Clear the file
    with open("./resources/mass_output.json", "w") as data_file:
        data_file.write("")
        data_file.flush()

    with open("./resources/mass_output.json", "a") as data_file:
        data_file.write("[\n")
        data_file.flush()

    for i in range(0, step_count):
        nn_data["lambda"] = _lambda
        top_data = get_best_model(run_count, nn_data)

        data = {
            "lambda": nn_data["lambda"],
            "performance": top_data[0],
            "weights": [x.tolist() for x in top_data[1]]
        }

        data_list.append(data)

        # Start new printer thread

        threads.append(JsonPrinterThread(to_write=data, path="./resources/mass_output.json"))

        threads[-1].start()

        print("Finished runs with lambda: " + str(_lambda))

        _lambda = _lambda + lambda_step

    for thread in threads:
        thread.join()

    with open("./resources/mass_output.json", "a") as data_file:
        data_file.write("]")
        data_file.flush()
