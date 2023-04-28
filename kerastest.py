import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense


# Generate dummy data
# x_train = np.random.random((1000, 20))
# y_train = np.random.randint(2, size=(1000, 1))
# x_test = np.random.random((100, 20))
# y_test = np.random.randint(2, size=(100, 1))

def trainKeras(x_train, y_train, x_test, y_test):

    print(x_train.shape)

    # x_train = x_train.astype(np.float32)
    # y_train = y_train.astype(np.float32)
    # x_test = x_test.astype(np.float32)
    # y_test = y_test.astype(np.float32)

    # Define the model
    model = Sequential()
    model.add(Dense(64, input_dim=8, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer='rmsprop',
                loss='binary_crossentropy',
                metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train,
            epochs=100,
            batch_size=32)

    # Evaluate the model
    score = model.evaluate(x_test, y_test, batch_size=32)