from numpy import mean, std
from datetime import datetime

from model import build_model, get_datasets
import matplotlib.pyplot as plt
import json


def run_tests():
    print("Beginning tests...")
    test_params()
    print("Tests finished! You can check results in 'resultsDD_MM_YYYY_HH_MM_SS.json' file.")


def test_params(save_file="results", sample_size=20, fln=[32], sln=[4], lr=[0.001], mom=[0.99], e=[20], bs=[20]):
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
    with open(file_name, "w") as file:
        json.dump(results, file, indent=4)

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


def plot_results(results):
    plt.plot(results)
    plt.show()


run_tests()