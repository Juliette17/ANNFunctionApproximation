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

from numpy import mean, std


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

    sns.lineplot(range(len(history['loss'])), history['loss'])
    plt.show()

    print("presenting evaluation results on training set, mse loss: ", train_results)

    print("presenting evaluation results on validation set, mse loss: ", results)
    y_pred = model.predict(X_valid)
    print("range of real z values: [%f, %f]" % (min(y_valid), max(y_valid)))

    error = [0] * len(y_pred)
    for j in range(len(y_pred)):
        error[j] = (y_pred[j] - y_valid[j]) ** 2

    print("Maximum square error: %.4f , minimum square error: %.4f" % (max(error), min(error)))
    print("Metric on validation_set: %.4f (%.4f) MSE (STD)" % (mean(error), std(error)))

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


run(train_new_model=False)
