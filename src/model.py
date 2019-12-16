from data_generator import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def prepare_datasets_for_training():
    df = load_dataset()
    # quick check on data
    print(df.head())
    print(df.describe())

    # y - value to predict (numpy array)
    # X - rest of data (pandas dataframe)
    y = df["z"].values 
    X = df.drop(["z"], axis = 1)

    # split dataset for train(60%), validation(20%) and test(20%)
    X_train_valid, X_test, y_train_valid, y_test = train_test_split(X, y, test_size=0.2, random_state=17)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid, y_train_valid, test_size=0.25, random_state=17)
    return X_train, X_valid, X_test, y_train, y_valid, y_test


def train_model(X_train, y_train, file_name_to_save_model = 'model.h5'):
    # define the keras model
    model = Sequential()
    model.add(Dense(5, input_dim=2, activation='relu'))
    model.add(Dense(1, activation='linear'))

    model.compile(loss='mean_squared_error',
                optimizer='adam')

    history = model.fit(X_train, y_train, epochs=10, batch_size=10)
    model.save('model.h5')

    return model, history

# X_test, y_test - to evaluate model only at the end of project!!!!!!!
X_train, X_valid, X_test, y_train, y_valid, y_test = prepare_datasets_for_training()
# uncomment to train new model
model, history = train_model(X_train, y_train)
model = load_model('model.h5')
results = model.evaluate(X_valid, y_valid)

#sns.lineplot(range(len(history.history['loss'])), history.history['loss'])
print("presenting evaluation results on validation set, mse loss: ", results)
y_pred = model.predict(X_valid)
print("range of real z values: [%f, %f]" % (min(y_valid), max(y_valid)))
# summarize the first 10 cases
for i in range(10):
	print('%d: %s => %f (expected %f)' % (i, X_valid.iloc[i,:].tolist(), y_pred[i], y_valid[i]))

