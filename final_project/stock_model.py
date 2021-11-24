import tensorflow as tf


class StockModel(tf.keras.Model):
    def __init__(self, batch_size):
        """
        The Reinforce class that inherits from tf.keras.Model
        The forward pass calculates the policy for the agent given a batch of states.

        :param state_size: number of parameters that define the state. You don't necessarily have to use this,
                           but it can be used as the input size for your first dense layer.
        :param num_actions: number of actions in an environment
        """
        super(StockModel, self).__init__()
        self.batch_size = batch_size
        self.learning_rate = 1e-3
        self.batch_size = batch_size

        self.price_prediction = tf.keras.Sequential(
            tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(self.batch_size, 1)),
            tf.keras.layers.LSTM(50, return_sequences=True),
            tf.keras.layers.LSTM(50, return_sequences=True),
            tf.keras.layers.Dense(1),  # linear_layer
        )

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    @tf.function
    def call(self, x_train):
        """
        call the function for a single training batch

        :param x_train: training batch
        :return:
        """
        return self.price_prediction(x_train)

    def loss(self, y_true, predictions):
        """
        Computes the loss using MSE between predict and actual price

        :param y_true: actual prices
        :param predictions: predictions from model
        :return: loss, a Tensorflow scalar
        """
        return tf.math.sqrt(tf.keras.losses.MeanSquaredError(y_true, predictions))
