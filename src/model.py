from data_generator import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

df = load_dataset()
# quick check on data
print(df.head())
print(df.describe())

# y - value to predict
# X - rest of data
y = df.iloc[:,2]
X = df.iloc[:,0:2]
#print()

# split dataset for train(60%), validation(20%) and test(20%)
X_train_valid, X_test, y_train_valid, y_test = train_test_split(X, y, test_size=0.2, random_state=17)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid, y_train_valid, test_size=0.25, random_state=17)

# define the keras model
model = Sequential()
model.add(Dense(5, input_dim=2, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=['mae'])
model.fit(X_train, y_train, epochs=10, batch_size=10)
results = model.evaluate(X_valid, y_valid)
print(results)