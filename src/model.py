from data_generator import load_dataset
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

from keras import optimizers

import pickle


def get_datasets(with_noise=False):
    x, y = prepare_dataset(with_noise)
    return split_dataset(x, y)


def prepare_dataset(with_noise=False):
    df = load_dataset(with_noise)
    dataset = df.values
    # quick check on data
    print(df.head())
    print(df.describe())
    # y - value to predict (numpy array)
    # X - rest of data (numpy array)
    X = dataset[:, 0:2]
    y = dataset[:, 2]
    return X, y


def split_dataset(X, y):
    # split dataset for train(60%), validation(20%) and test(20%)
    X_train_valid, X_test, y_train_valid, y_test = train_test_split(X, y, test_size=0.2, random_state=67)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid, y_train_valid, test_size=0.25, random_state=67)
    return X_train, X_valid, X_test, y_train, y_valid, y_test


def train_model(X_train, y_train, model_file='model.h5', history_file='trainHistoryDict'):
    model = build_model()
    history = model.fit(X_train, y_train, epochs=20, batch_size=10)
    model.save(model_file)
    with open(history_file, 'wb') as file:
        pickle.dump(history.history, file)
    return model, history.history


def build_model(layers_neurons=[32,8,1], lr=0.002, momentum=0.99):
    # define the keras model
    model = Sequential()
    for i in range(len(layers_neurons)):
        if i == 0:
            model.add(Dense(layers_neurons[i], input_dim=2, kernel_initializer='normal', activation='tanh'))
        elif i == len(layers_neurons)-1:
            model.add(Dense(layers_neurons[i], kernel_initializer='normal'))
        else:
            model.add(Dense(layers_neurons[i], kernel_initializer='normal', activation='tanh'))

    sgd = optimizers.SGD(lr=lr, decay=1e-6, momentum=momentum, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd)

    return model





