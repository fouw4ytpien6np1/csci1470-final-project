import tensorflow as tf


def create_lstm_model(x_train, y_train, x_test, batch_size, learning_rate, epochs):
    """
    create LSTM model trained on x_train and y_train
    and make predictions on the x_test data

    :param x_train: x train data in shape (len(train_data), batch_size)
    :param y_train: y train data in shape (len(train_data), 1)
    :param x_test: x test data in shape (len(test_data), batch_size)
    :param batch_size: batch size
    :param epochs: number of training epochs
    :param learning_rate: optimizer learning rate
    :return: lstm, predicted_stock_price (unscaled)
    """
    # cnn architecture:
    cnn = tf.keras.Sequential(
        [
            tf.keras.layers.Conv1D(4, kernel_size=1, activation="tanh", padding='same'),
            tf.keras.layers.MaxPooling1D(pool_size=1, padding="same")
        ]
    )

    cnn_output = cnn(x_train)

    # The lstm architecture
    cnn_lstm = tf.keras.Sequential()
    # Add layers
    cnn_lstm.add(tf.keras.layers.LSTM(units=50, return_sequences=True))
    cnn_lstm.add(tf.keras.layers.LSTM(units=50, return_sequences=True))
    cnn_lstm.add(tf.keras.layers.LSTM(units=50, return_sequences=True))
    cnn_lstm.add(tf.keras.layers.LSTM(units=50, activation='tanh'))
    cnn_lstm.add(tf.keras.layers.Dense(units=4))

    # Compiling the GRU
    cnn_lstm.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                     loss='mean_absolute_error')
    # Fitting to the training set
    cnn_lstm.fit(cnn_output, y_train, epochs=epochs, batch_size=batch_size)

    cnn_x_test_output = cnn(x_test)
    predicted_stock_price = cnn_lstm.predict(cnn_x_test_output)

    return cnn_lstm, predicted_stock_price
