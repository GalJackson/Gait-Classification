import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten, Dropout


def generate_model(input_shape):
    model = Sequential()

    # two LSTM layers of 64 units
    model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(64, return_sequences=True))

    # 3 CNN 1D layers with varying numbers of 1x3 convolution kernels, separated by pooling layers 
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2, strides=2))
    model.add(Conv1D(filters=144, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2, strides=2))
    model.add(Conv1D(filters=156, kernel_size=3, activation='relu'))

    # flatten the output in order to input it to a fully connected layer 
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))

    # a dropout layer of 0.5 is used to help prevent overfitting
    model.add(Dropout(0.5))

    # output layer
    # model.add(Dense(2, activation='softmax'))
    model.add(Dense(1, activation='sigmoid'))

    # compile the model to optimise using the Adam optimiser
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model



if __name__ == "__main__":
    # set the seed for result reproducability
    tf.random.set_seed(41)

    # size of each data instance
    input_shape = (150, 20)

    # initialise the LSTM-CNN model
    model = generate_model(input_shape)