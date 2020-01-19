from numpy import mean

from model import build_model, get_datasets
import numpy as np
import matplotlib.pyplot as plt
import json
from tensorflow import set_random_seed

# TODO: test of batch size
# TODO: test on validation set
# TODO: test at the end with the best params on test set
# TODO: test on noized data
def run_tests():
    print("Beginning tests...")
    #results = test_first_layer_size()
    #plot_results(results)
    #results = test_second_layer_size()
    #plot_results(results)
    #results = test_learning_rate()
    #results = test_learning_rate([0.0020, 0.0015, 0.0013, 0.001, 0.0008, 0.0005])
    #results = test_momentum()
    #results = test_epochs()
    print("Tests finished! You can check results in 'results.json' file.")


# TODO: evaluate on validation set
def test_first_layer_size(max_size=6):
    X_train, X_valid, X_test, y_train, y_valid, y_test = get_datasets()
    results = {}
    for i in range(max_size):
        temp_results = []
        for j in range(20):
            model = build_model(layers_neurons=[2**i, 1])
            history = model.fit(X_train, y_train, epochs=20, batch_size=10)
            temp_results.append(history.history["loss"][-1])
        results[2**i] = mean(temp_results)

    with open('results.json', "r") as file:
        data = json.load(file)

    data["first_layer_size"] = results

    with open('results.json', "w") as file:
        json.dump(data, file, indent=4)
    return results


# TODO: evaluate on validation set
def test_second_layer_size(max_size=6):
    X_train, X_valid, X_test, y_train, y_valid, y_test = get_datasets()
    results = {}
    for i in range(max_size):
        temp_results = []
        for j in range(20):
            model = build_model(layers_neurons=[2**(max_size-1), 2**i, 1])
            history = model.fit(X_train, y_train, epochs=20, batch_size=10)
            temp_results.append(history.history["loss"][-1])
        results[2**i] = mean(temp_results)

    with open('results.json', "r") as file:
        data = json.load(file)

    data["second_layer_size"] = results

    with open('results.json', "w") as file:
        json.dump(data, file, indent=4)
    return results

# TODO: evaluate on validation set
def test_learning_rate(lrs=[0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]):
    X_train, X_valid, X_test, y_train, y_valid, y_test = get_datasets()
    results = {}
    for i in range(len(lrs)):
        temp_results = []
        for j in range(20):
            model = build_model(layers_neurons=[32, 4, 1], lr=lrs[i])
            history = model.fit(X_train, y_train, epochs=20, batch_size=10)
            temp_results.append(history.history["loss"][-1])
        results[lrs[i]] = mean(temp_results)

    with open('results.json', "r") as file:
        data = json.load(file)

    data["learning_rate"] = results

    with open('results.json', "w") as file:
        json.dump(data, file, indent=4)
    return results

# TODO: evaluate on validation set
def test_momentum(m=[0.99, 0.97, 0.95, 0.93, 0.91]):
    X_train, X_valid, X_test, y_train, y_valid, y_test = get_datasets()
    results = {}
    for i in range(len(m)):
        temp_results = []
        for j in range(20):
            model = build_model(layers_neurons=[32, 4, 1], lr=0.001, momentum=m[i])
            history = model.fit(X_train, y_train, epochs=20, batch_size=10)
            temp_results.append(history.history["loss"][-1])
        results[m[i]] = mean(temp_results)

    with open('results.json', "r") as file:
        data = json.load(file)

    data["momentum"] = results

    with open('results.json', "w") as file:
        json.dump(data, file, indent=4)
    return results


def test_epochs(e=[10, 12, 14, 16, 18, 20, 22, 24, 26]):
    X_train, X_valid, X_test, y_train, y_valid, y_test = get_datasets()
    results = {}
    results_validation = {}
    for i in range(len(e)):
        temp_results = []
        temp_results2 = []
        for j in range(20):
            model = build_model(layers_neurons=[32, 4, 1], lr=0.001, momentum=0.99)
            history = model.fit(X_train, y_train, epochs=e[i], batch_size=10)
            temp_results.append(history.history["loss"][-1])
            temp_results2.append(model.evaluate(X_valid, y_valid))
        results[e[i]] = mean(temp_results)
        results_validation[e[i]] = mean(temp_results2)

    with open('results.json', "r") as file:
        data = json.load(file)

    data["epochs"] = results
    data["epochs_valid"] = results_validation

    with open('results.json', "w") as file:
        json.dump(data, file, indent=4)
    return results


def plot_results(results):
    plt.plot(results)
    plt.show()


run_tests()
