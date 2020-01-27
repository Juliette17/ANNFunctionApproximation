from model import get_datasets, train_model, build_model, prepare_dataset
from data_generator import plot_surface

import pickle
from keras.models import load_model
import seaborn as sns
import matplotlib.pyplot as plt

from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from numpy import mean, std, arange
import pandas as pd
from datetime import datetime


def evaluate_model_on_noisy_data(train_models=False, plot_history=False):
    X_train, X_valid, X_test, y_train, y_valid, y_test = get_datasets()
    X_train_noise, X_valid_noise, X_test_noise, y_train_noise, y_valid_noise, y_test_noise = get_datasets(
        with_noise=True)

    dt = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

    if train_models:
        model, history = train_model(X_train, y_train, "model{}.h5".format(dt), "trainHistoryDict{}".format(dt))
        noisy_model, noisy_history = train_model(X_train_noise, y_train_noise, "noisy_model{}.h5".format(dt), "noisy_trainHistoryDict{}".format(dt))
    else:
        model = load_model("model27_01_2020_16_45_06.h5")
        noisy_model = load_model("noisy_model27_01_2020_16_45_06.h5")
        with open("trainHistoryDict27_01_2020_16_45_06", "rb") as file:
            history = pickle.load(file)
        with open("noisy_trainHistoryDict27_01_2020_16_45_06", "rb") as file:
            noisy_history = pickle.load(file)

    if plot_history:
        plot_loss_history(history, "loss_during_training{}".format(dt))
        plot_loss_history(noisy_history, "loss_during_training_noise{}".format(dt), with_noise=True)

    metrics = []
    metrics.append(["Train"] + calculate_metrics(model.predict(X_train), y_train))
    metrics.append(["Validation"] + calculate_metrics(model.predict(X_valid), y_valid))
    metrics.append(["Test"] + calculate_metrics(model.predict(X_test), y_test))
    metrics.append(["Train with noise"] + calculate_metrics(noisy_model.predict(X_train_noise), y_train_noise))
    metrics.append(["Validation with noise"] + calculate_metrics(noisy_model.predict(X_valid_noise), y_valid_noise))
    metrics.append(["Test with noise"] + calculate_metrics(model.predict(X_test_noise), y_test_noise))

    save_metrics(dt, metrics)


def plot_loss_history(history, file_name, with_noise=False):
    plt.plot(range(2, len(history['loss'])), history['loss'][2:], label="Train set with noise" if with_noise else "Train set")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Loss during training.")
    plt.savefig(file_name)
    plt.show()


def calculate_metrics(y_pred, y):
    errors = [(y_pred[j][0] - y[j]) ** 2 for j in range(len(y_pred))]
    result = [min(errors), max(errors), mean(errors), std(errors)]
    return [round(r, 7) for r in result]


def save_metrics(dt, metrics):
    df = pd.DataFrame.from_records(metrics)
    df.columns = ['Set', 'Min MSE', 'Max MSE', 'Mean MSE', 'Stand. Dev. MSE']
    df.to_csv('metrics{}.csv'.format(dt), index=False)

    plot_metrics(df[:][0:3], dt, with_noise=False)
    plot_metrics(df[:][3:6], dt, with_noise=True)


def plot_metrics(df, dt, with_noise):
    means = df['Mean MSE']
    stdev = df['Stand. Dev. MSE']
    ind = arange(3)
    width = 0.35
    plt.bar(ind - width/2, means, width, label='Mean')
    plt.bar(ind + width/2, stdev, width, label='Standard deviation')
    plt.ylabel('MSE')
    plt.title('MSE by sets with noise' if with_noise else 'MSE by sets without noise')
    plt.xticks(ind, df['Set'])
    plt.yticks()
    plt.legend()
    plt.savefig("metrics{}.png".format(dt))
    plt.show()


# TODO: add ArgumentParser
def run(train_new_model=False, model_file='model.h5', history_file='trainHistoryDict', evaluation='manual'):
    X_train, X_valid, X_test, y_train, y_valid, y_test = get_datasets()

    if evaluation == 'keras':
        evaluate_with_keras_pipeline(X_valid, y_valid)
    else:
        if train_new_model:
            model, history = train_model(X_train, y_train, model_file, history_file)
        else:
            model = load_model(model_file)
            with open(history_file, "rb") as file:
                history = pickle.load(file)

        evaluate_manually(X_train, y_train, X_valid, y_valid, model, history)

    return model, history


def evaluate_manually(X_train, y_train, X_valid, y_valid, model, history):
    # X_test, y_test - to evaluate model only at the end of project!!!!!!!
    train_results = model.evaluate(X_train, y_train)
    results = model.evaluate(X_valid, y_valid)

    sns.lineplot(range(2, len(history['loss'])), history['loss'][2:])
    plt.show()

    print("presenting evaluation results on training set, mse loss: %.6f" % train_results)

    print("presenting evaluation results on validation set, mse loss: %.6f" % results)
    y_pred = model.predict(X_valid)
    print("range of real z values: [%f, %f]" % (min(y_valid), max(y_valid)))

    error = [(y_pred[j] - y_valid[j]) ** 2 for j in range(len(y_pred))]

    print("Maximum square error: %.6f , minimum square error: %.6f" % (max(error), min(error)))
    print("Metric on validation_set: %.6f (%.6f) MSE (STD)" % (mean(error), std(error)))

    plot_surface(X_valid[:, 0], X_valid[:, 1], y_pred[:, 0])


def evaluate_with_keras_pipeline(X_valid, y_valid):
    # evaluate model with standardized dataset
    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('mlp', KerasRegressor(build_fn=build_model, epochs=20, batch_size=100, verbose=0)))
    pipeline = Pipeline(estimators)
    kfold = KFold(n_splits=4)
    X, Y = prepare_dataset()
    results = cross_val_score(pipeline, X, Y, cv=kfold)
    print("Larger: %.4f (%.4f) MSE" % (results.mean(), results.std()))


#run(train_new_model=True)
evaluate_model_on_noisy_data(train_models=False)