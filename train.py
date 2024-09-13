import os 
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten, Dropout
from sklearn.model_selection import train_test_split, GridSearchCV
import tkinter
from tkinter import filedialog
from sklearn.metrics import classification_report, confusion_matrix
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.optimizers import Adam


def generate_lstm_cnn_model(input_shape, lstm_units=64, dense_units=128, lstm_layers=2, conv_filters=156, kernel_size=3, learning_rate=0.005):
    model = Sequential()

    # LSTM layers - default to two LSTM layers of 64 units
    model.add(LSTM(lstm_units, input_shape=input_shape, return_sequences=True))
    for _ in range(lstm_layers - 1):
        model.add(LSTM(lstm_units, return_sequences=True))

    # 3 CNN 1D layers with varying numbers of 1 x kernel_size convolution kernels, separated by pooling layers 
    model.add(Conv1D(filters=128, kernel_size=kernel_size, activation='relu'))
    model.add(MaxPooling1D(pool_size=2, strides=2))
    model.add(Conv1D(filters=144, kernel_size=kernel_size, activation='relu'))
    model.add(MaxPooling1D(pool_size=2, strides=2))
    model.add(Conv1D(filters=conv_filters, kernel_size=kernel_size, activation='relu'))

    # flatten the output in order to input it to a fully connected layer 
    model.add(Flatten())
    model.add(Dense(dense_units, activation='relu'))

    # a dropout layer of 0.5 is used to help prevent overfitting
    model.add(Dropout(0.5))

    # output layer
    model.add(Dense(1, activation='sigmoid'))

    # compile the model to optimise using the Adam optimiser
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model


def single_model_training(X_train, X_test, Y_train, Y_test, input_shape):
    # initialise the LSTM-CNN model
    model = generate_lstm_cnn_model(input_shape)
    model.summary()

    # train model 
    history = model.fit(X_train, Y_train, epochs=50, batch_size=128, validation_split=0.2)

    # test model
    test_loss, test_accuracy = model.evaluate(X_test, Y_test)
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int)  
    print(f'Accuracy: {test_accuracy:.2f}')
    print(confusion_matrix(Y_test, y_pred))
    print(classification_report(Y_test, y_pred))

    print("Training complete.")


def parameter_tuning_grid(input_shape):
    """ Returns a GridSearch parameter tuning grid for the LSTM-CNN model"""

    param_grid = {
        'lstm_units': [32, 64, 128],
        'dense_units': [32, 64, 96, 128],
        'lstm_layers': [2, 3],
        'conv_filters': [144, 156],
        'kernel_size': [2, 3, 4],
        'learning_rate': [0.005, 0.01]
    }

    model = KerasClassifier(
        model=generate_lstm_cnn_model, 
        input_shape=input_shape, 
        epochs=25, 
        batch_size=128, 
        verbose=1,
        lstm_units=64, 
        dense_units=128, 
        lstm_layers=2, 
        conv_filters=156, 
        kernel_size=3,
        learning_rate=0.005
    )

    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1)

    return grid


def prepare_data(input_shape):
    """ 
    Load and prepare CSV files for model training
    Returns the data in a test-train split, formatted as X_train, X_test, Y_train, Y_test
    """
    # set the seed for result reproducability
    tf.random.set_seed(51)

    # initialize lists for input data
    x_abnormal = []
    x_normal = []

    # pop-up window to select input directory 
    tkinter.Tk().withdraw() # prevents an empty tkinter window from appearing
    input_dir = filedialog.askdirectory(title="Select input directory")

    abnormal_prefixes = ['OAW02', 'OAW12', 'OAW13'] # all files starting with these strings have POMA score of < 12, abnormal gait

    # load data
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.csv'):
            # load csv into a df
            file_path = os.path.join(input_dir, file_name)
            df = pd.read_csv(file_path, index_col=0)
            
            # save data into the appropriate class
            if df.shape == input_shape:
                if any(file_name.startswith(prefix) for prefix in abnormal_prefixes):
                    x_abnormal.append(df.values)
                else:
                    x_normal.append(df.values)

    # convert input data to 3D numpy arrays
    x_abnormal = np.array(x_abnormal)
    x_normal = np.array(x_normal)

    # randomize the order of each dataset
    np.random.shuffle(x_abnormal)
    np.random.shuffle(x_normal)

    # balance dataset to have a 50/50 split of normal vs abnormal gait 
    min_size = min(len(x_abnormal), len(x_normal))
    x_abnormal = x_abnormal[:min_size]
    x_normal = x_normal[:min_size]

    X = np.concatenate((x_abnormal, x_normal), axis=0)
    Y = np.array([1] * len(x_abnormal) + [0] * len(x_normal))  # 1 for abnormal, 0 for normal

    # shuffle again to avoid bias and ensure randomness in cross validation
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]

    # test-train split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=51, stratify=Y)

    return X_train, X_test, Y_train, Y_test


def main():
    # shape of each data instance
    input_shape = (150, 16)

    # Load all training and testing data
    X_train, X_test, Y_train, Y_test = prepare_data(input_shape)

    # generate hyperparameter tuning grid
    grid = parameter_tuning_grid(input_shape)

    # fit the training data on parameter tuning grid
    grid_result = grid.fit(X_train, Y_train)
    print(f"Best model accuracy: {grid_result.best_score_} using parameters: {grid_result.best_params_}")

    # evaluate best model on test data
    Y_pred = grid.predict(X_test)
    print(classification_report(Y_test, Y_pred))
    print(confusion_matrix(Y_test, Y_pred))

if __name__ == "__main__":
    main()