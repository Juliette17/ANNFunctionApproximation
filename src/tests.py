from model import build_model, get_datasets
import numpy as np
import matplotlib.pyplot as plt
import json

# TODO: add test of learning_rate
# TODO: add test of momentum
# TODO: add test of number of epochs
# TODO: add test of batch size
def run_tests():
    results = test_first_layer_size()
    plot_results(results)
    results = test_second_layer_size()
    plot_results(results)

# TODO: evaluate on validation set
def test_first_layer_size(max_size=6):
    X_train, X_valid, X_test, y_train, y_valid, y_test = get_datasets()
    results = np.zeros((max_size))
    for i in range(max_size):
        model = build_model(layers_neurons=[2**i, 1])
        history = model.fit(X_train, y_train, epochs=20, batch_size=10)
        results[i] = history.history["loss"][-1]

    with open('results.json', "r") as file:
        data = json.load(file)

    data["first_layer_size"] = list(results)

    with open('results.json', "w") as file:
        json.dump(data, file, indent=4)
    return results

# TODO: evaluate on validation set
def test_second_layer_size(max_size=6):
    X_train, X_valid, X_test, y_train, y_valid, y_test = get_datasets()
    results = np.zeros((max_size))
    for i in range(max_size):
        model = build_model(layers_neurons=[2**(max_size-1), 2**i, 1])
        history = model.fit(X_train, y_train, epochs=20, batch_size=10)
        results[i] = history.history["loss"][-1]

    with open('results.json', "r") as file:
        data = json.load(file)

    data["second_layer_size"] = list(results)

    with open('results.json', "w") as file:
        json.dump(data, file, indent=4)
    return results


def plot_results(results):
    plt.plot(results)
    plt.show()


run_tests()
