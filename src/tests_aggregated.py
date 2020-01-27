from numpy import mean, std
from datetime import datetime

from model import build_model, get_datasets
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import json

import re


def run_tests():
    print("Beginning tests...")
    start = timer()
    test_params(
        param_to_test="first_layer_neurons",
        sample_size=10,
        fln=[2, 4, 8, 16, 32]
    )
    test_params(
        param_to_test="second_layer_neurons",
        sample_size=10,
        sln=[2, 4, 8, 16, 32]
    )
    test_params(
        param_to_test="learning_rate",
        sample_size=10,
        lr=[0.01, 0.005, 0.002, 0.001, 0.0008, 0.0005, 0.0001]
    )
    test_params(
        param_to_test="momentum",
        sample_size=10,
        mom=[0.9, 0.925, 0.95, 0.975, 0.99]
    )
    test_params(
        param_to_test="epochs",
        sample_size=10,
        e=[12, 14, 16, 18, 20, 22, 24, 26]
    )
    test_params(
        param_to_test="batch_size",
        sample_size=10,
        bs=[1, 5, 10, 20, 50, 100]
    )
    end = timer()
    print("Tests finished with {} seconds! You can check results in 'resultsDD_MM_YYYY_HH_MM_SS.json' file.".format(end - start))


def test_params(param_to_test, save_file="results", sample_size=20, fln=[32], sln=[4], lr=[0.001], mom=[0.99], e=[20], bs=[20]):
    X_train, X_valid, X_test, y_train, y_valid, y_test = get_datasets()
    results = []
    for i in range(len(fln)):
        for j in range(len(sln)):
            for k in range(len(lr)):
                for l in range(len(mom)):
                    for m in range(len(e)):
                        for n in range(len(bs)):
                            train_mean, train_std, valid_mean, valid_std = test(
                                X_train, y_train, X_valid, y_valid,
                                sample=sample_size,
                                fln=fln[i], sln=sln[j], lr=lr[k], m=mom[l], e=e[m], bs=bs[n])
                            result = {
                                "first_layer_neurons": fln[i],
                                "second_layer_neurons": sln[j],
                                "learning_rate": lr[k],
                                "momentum": mom[l],
                                "epochs": e[m],
                                "batch_size": bs[n],
                                "sample_size": sample_size,
                                "train_mean": train_mean,
                                "train_std": train_std,
                                "valid_mean": valid_mean,
                                "valid_std": valid_std
                            }
                            results.append(result)

    file_name = save_file + datetime.now().strftime("%d_%m_%Y_%H_%M_%S") + ".json"
    data = {param_to_test: results}

    with open(file_name, "w") as file:
        json.dump(data, file, indent=4)

    return results


def test(X_train, y_train, X_valid, y_valid, sample=20, fln=32, sln=4, lr=0.001, m=0.99, e=20, bs=10):
    train_results = []
    valid_results = []
    for j in range(sample):
        model = build_model(layers_neurons=[fln, sln, 1], lr=lr, momentum=m)
        history = model.fit(X_train, y_train, epochs=e, batch_size=bs)

        train_results.append(history.history["loss"][-1])
        valid_results.append(model.evaluate(X_valid, y_valid))

    return mean(train_results), std(train_results), mean(valid_results), std(valid_results)


def generate_plots():
    for i in range(1, 7):
        file_name = "parameters_tests//results26_01_2020_{}.json".format(i)
        with open(file_name, "r") as file:
            data = json.load(file)

        param_values, loss_values = get_values_for_plot(data)
        plot_results(param_values, loss_values, list(data)[0])


def generate_plots_noise():
    for i in range(6, 7):
        file_name = "parameters_tests_noise//results27_01_2020_{}.json".format(i)
        with open(file_name, "r") as file:
            data = json.load(file)

        param_values, loss_values = get_values_for_plot(data)
        plot_results(param_values, loss_values, list(data)[0])


def get_values_for_plot(data):
    key = list(data)[0]
    param_values = []
    loss_values = {"train_mean": [], "train_std": [], "valid_mean": [], "valid_std": []}
    for test_instance in data[key]:  # experiment instance for every value of param
        param_values.append(test_instance[key])
        for loss in list(loss_values):  # train_mean/std, valid_mean/std
            loss_values[loss].append(test_instance[loss])
    return param_values, loss_values


def plot_results(param_values, loss_values, parameter):
    parameter_for_label = parameter.replace("_", " ")
    fig = plt.figure()
    for key in loss_values.keys():
        plt.plot(param_values[2:], loss_values[key][2:], label=key)
    plt.legend()
    plt.xlabel(parameter_for_label)
    plt.ylabel("loss")
    plt.title("Loss on train and validation set for different values of {}.".format(parameter_for_label))
    plt.savefig("loss_for_{}_2.png".format(parameter))
    #plt.show()


#run_tests()
generate_plots()
#generate_plots_noise()
