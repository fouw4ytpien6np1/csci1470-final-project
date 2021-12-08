import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import matplotlib.pyplot as plt
from random import randrange

from final_project.lstm_stock_model.stock_model import StockModel
from final_project.utils.candle_stick_graph import create_candlestick_graph

# see this article for the metrics below:
# https://towardsdatascience.com/time-series-forecast-error-metrics-you-should-know-cc88b8c67f27


# calculate Root Mean Squared Error (RMSE)
def rmse(y, y_hat):
    return np.sqrt(np.mean(np.square(y - y_hat)))


# calculate Mean Absolute Percentage Error (MAPE)
def mape(y, y_hat):
    return np.mean(np.abs((y - y_hat) / y) * 100)


# SMAPE proposed by Makridakis (1993): 0%-200%
def smape_original(a, f):
    return 1/len(a) * np.sum(2 * np.abs(f-a) / (np.abs(a) + np.abs(f))*100)


# adjusted SMAPE version to scale metric from 0%-100%
def smape_adjusted(a, f):
    return (1/a.size * np.sum(np.abs(f-a) / (np.abs(a) + np.abs(f))*100))


def batch_data(data, batch_size=1):
    """
    batches data

    :param data: data to batch
    :param batch_size: size of the batch
    :return: np.array(dataX), np.array(dataY)
    """

    dataX, dataY = [], []

    overall_length = len(data)

    for i in range(0, overall_length - (batch_size + 1)):

        if i == overall_length - (batch_size + 1):
            y = data[i + 1: i + 1 + batch_size]
            assert y == data[-100]
        x = data[i: i + batch_size]
        y = data[i + 1: i + 1 + batch_size]

        if len(y) == batch_size and len(x) == batch_size:
            dataX.append(x)
            dataY.append(y)

    return np.array(dataX), np.array(dataY)


def custom_accuracy(y_true, predictions):
    assert len(y_true) == len(predictions)
    y_true_up = []
    y_predictions_up = []
    for i in range(1, len(y_true)):
        if y_true[i] > y_true[i - 1]:
            y_true_up.append(1)
        else:
            y_true_up.append(0)

        if predictions[i] > predictions[i - 1]:
            y_predictions_up.append(1)
        else:
            y_predictions_up.append(0)

    assert len(y_true_up) == len(y_true) - 1
    assert len(y_predictions_up) == len(predictions) - 1
    return tf.convert_to_tensor(y_true_up), tf.convert_to_tensor(y_predictions_up)


def train(model, x_train, y_train):
    """
    trains the model for 1 epoch

    :param model: model to train
    :param x_train: the x training data
    :param y_train: the y training data
    :return: None

    """
    for i in range(len(x_train)):
        with tf.GradientTape() as tape:
            predictions = model.call(x_train[i])
            predictions = tf.reshape(predictions, shape=(model.batch_size, 1))
            loss = model.loss(y_train[i], predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def test(model, x_test, y_true, scaler):
    """
    tests the trained model

    :param model: trained model to test
    :param x_test: the x test data
    :return: avg_precision, avg_recall, avg_accuracy, avg_f_measure, correct_test_predictions
    """
    correct_predictions = []

    for i in range(0, len(x_test)):
        test_predictions = model.call(x_test[i])
        # add price prediction (as actual prices) to output list
        correct_predictions.append(test_predictions[0][-1])

    y_true = np.array(y_true)
    y_true_scaled = scaler.inverse_transform(y_true)
    correct_predictions = np.array(correct_predictions).reshape(-1, 1)

    correct_predictions_scaled = scaler.inverse_transform(correct_predictions)

    # Run Price Specific Measures
    lstm_rmse = rmse(y_true_scaled, correct_predictions_scaled)
    lstm_mape = mape(y_true_scaled, correct_predictions_scaled)
    lstm_smape = smape_original(y_true_scaled, correct_predictions_scaled)
    lstm_smape_adjusted = smape_adjusted(y_true_scaled, correct_predictions_scaled)

    # run Precision, Recall, Accuracy, and F-Measure on model's ability to predict
    # the next day is up or down

    y_true_up, y_predictions_up = custom_accuracy(y_true, correct_predictions)

    # calculate precision:
    precision_metric = tf.keras.metrics.Precision()
    precision_metric.update_state(y_pred=y_predictions_up, y_true=y_true_up)
    precision = precision_metric.result().numpy()

    # calculate recall:
    recall_metric = tf.keras.metrics.Recall()
    recall_metric.update_state(y_pred=y_predictions_up, y_true=y_true_up)
    recall = recall_metric.result().numpy()

    # calculate accuracy
    accuracy_metric = tf.keras.metrics.Accuracy()
    accuracy_metric.update_state(y_pred=y_predictions_up, y_true=y_true_up)
    accuracy = accuracy_metric.result().numpy()

    # calculate f_measure
    f_measure = 2 * precision * recall / (precision + recall)

    return lstm_rmse, lstm_mape, lstm_smape, lstm_smape_adjusted, \
           precision, recall, accuracy, f_measure, correct_predictions


def visualize_results(real_stock_price, predicted_stock_price, train_or_test, fig_num):
    plt.figure(fig_num)
    plt.plot(real_stock_price, color='black', label='Stock Price')
    plt.plot(predicted_stock_price, color='green', label='Predicted Stock Price')
    plt.title('Stock Price Prediction: ' + train_or_test)
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()
    rand_int = randrange(10000)
    plt.savefig('../../data/images/fig' + str(rand_int) + '.png')


def run_model(data, price_point, batch_size):
    # reset the index of the dataframe:
    reset_data = data.reset_index()[price_point]

    # LSTM sensitive to data scale, so normalize:
    scaler = MinMaxScaler(feature_range=(0, 1))

    # fit the data with the scaler:
    fit_data = scaler.fit_transform(np.array(reset_data).reshape(-1, 1))

    # get our training size (65% of the data):
    training_size = int(len(fit_data) * 0.65)

    # get our data by train and test:
    train_data = fit_data[0: training_size]
    test_data = fit_data[training_size: len(fit_data)]

    # BEGIN MODEL

    # init our model:
    model = StockModel(batch_size=batch_size)

    # batch the train data:
    x_train, y_train = batch_data(train_data, model.batch_size)

    # batch the test data:
    x_test, y_test = batch_data(test_data, model.batch_size)
    y_test_single_vals = []
    for array in y_test:
        y_test_single_vals.append(array[-1])

    test_data_without_scale = \
        scaler.inverse_transform(np.array(y_test_single_vals).reshape(-1, 1))

    # train the model:
    NUM_EPOCHS = 1

    for epoch in range(NUM_EPOCHS):
        train(model, x_train, y_train)
        model.reset_states()

    # run model on test
    lstm_rmse, lstm_mape, lstm_smape, lstm_smape_adjusted, precision, recall, accuracy, f_measure, predictions = \
        test(model, x_test, y_test_single_vals, scaler)

    predictions = np.array(predictions).reshape(-1, 1)
    predictions_without_scale = scaler.inverse_transform(predictions)

    print("###LSTM-Only TEST RESULTS for " + price_point + " ###")
    print("-------------------")
    print("Test Root Mean Squared Error: " + str(lstm_rmse))
    print("-------------------")
    print("Test Mean Absolute Percentage Error: " + str(lstm_mape))
    print("-------------------")
    print("Test Symmetric Mean Absolute Percentage Error: " + str(lstm_smape))
    print("-------------------")
    print("Test Adjusted Symmetric Mean Absolute Percentage Error: " + str(lstm_smape_adjusted))
    print("-------------------")
    print("Test Precision: " + str(precision))
    print("-------------------")
    print("Test Recall: " + str(recall))
    print("-------------------")
    print("Test Accuracy: " + str(accuracy))
    print("-------------------")
    print("Test F-Measure: " + str(f_measure))
    print("-------------------")
    print("###END OF RESULTS###")

    title = "TEST PREDICTIONS for " + price_point + " (LSTM Only Model)"

    visualize_results(test_data_without_scale, predictions_without_scale, title, 0)

    return predictions_without_scale


def main():
    # PREPROCESS
    BATCH_SIZE = 100

    # get data from CSV:
    data = pd.read_csv('../../data/spx_prices.csv')

    # reverse data
    data = data.iloc[::-1]

    training_size = int(len(data) * 0.65)

    # Date:
    dates = data.reset_index()["Date"]
    test_dates = dates[training_size: len(dates)]
    dates_for_charts = test_dates[BATCH_SIZE: len(test_dates)].reset_index()["Date"]

    # Open:
    real_open = data.reset_index()["Open"]
    test_open = real_open[training_size: len(real_open)]
    open_for_chart = test_open[BATCH_SIZE: len(test_open)].reset_index()["Open"]

    # High
    real_high = data.reset_index()["High"]
    test_high = real_high[training_size: len(real_high)]
    high_for_chart = test_high[BATCH_SIZE: len(test_high)].reset_index()["High"]

    # High
    real_low = data.reset_index()["Low"]
    test_low = real_low[training_size: len(real_low)]
    low_for_chart = test_low[BATCH_SIZE: len(test_low)].reset_index()["Low"]

    # High
    real_close = data.reset_index()["Close"]
    test_close = real_close[training_size: len(real_close)]
    close_for_chart = test_close[BATCH_SIZE: len(test_close)].reset_index()["Close"]

    title1 = "SPX Actual Candles for Test Data"

    create_candlestick_graph(dates_for_charts, open_for_chart, high_for_chart, low_for_chart, close_for_chart, title1)

    price_points = ["Open", "High", "Low", "Close"]
    predicted_prices = []

    for val in price_points:
        predicted_price = run_model(data, val, BATCH_SIZE)
        predicted_prices.append(predicted_price)

    predicted_open = predicted_prices[0]
    predicted_high = predicted_prices[1]
    predicted_low = predicted_prices[2]
    predicted_close = predicted_prices[3]

    df = pd.DataFrame(columns=["Open", "High", "Low", "Close"])
    df["Open"] = np.reshape(predicted_open, newshape=(len(predicted_open,)))
    df["High"] = np.reshape(predicted_high, newshape=(len(predicted_high,)))
    df["Low"] = np.reshape(predicted_low, newshape=(len(predicted_low,)))
    df["Close"] = np.reshape(predicted_close, newshape=(len(predicted_close,)))

    title2 = "SPX Predicted Candles using LSTM Only Model on Test"

    create_candlestick_graph(dates_for_charts, df["Open"], df["High"], df["Low"], df["Close"], title2)


if __name__ == '__main__':
    main()
