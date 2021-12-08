import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from final_project.lstm_stock_model.main import visualize_results
from final_project.cnn_lstm_model.model import create_lstm_model


# see this article for the metrics below:
# https://towardsdatascience.com/time-series-forecast-error-metrics-you-should-know-cc88b8c67f27


# calculate Root Mean Squared Error (RMSE)
from final_project.utils.candle_stick_graph import create_candlestick_graph


def rmse(y, y_hat):
    return np.sqrt(np.mean(np.square(y - y_hat)))


# calculate Mean Absolute Percentage Error (MAPE)
def mape(y, y_hat):
    return np.mean(np.abs((y - y_hat) / y) * 100)


# SMAPE proposed by Makridakis (1993): 0%-200%
def smape_original(a, f):
    return 1 / len(a) * np.sum(2 * np.abs(f - a) / (np.abs(a) + np.abs(f)) * 100)


# adjusted SMAPE version to scale metric from 0%-100%
def smape_adjusted(a, f):
    return (1 / a.size * np.sum(np.abs(f - a) / (np.abs(a) + np.abs(f)) * 100))


def batch_data(data, batch_size=1):
    """
    batches data

    :param data: data to batch
    :param batch_size: size of the batch
    :return:np.array(dataX), np.array(dataY)
    """

    dataX, dataY = [], []

    overall_length = len(data)

    for i in range(0, overall_length - (overall_length % batch_size)):
        x = data[i: i + batch_size]

        if i + batch_size == overall_length:
            break
        else:
            y = data[i + batch_size]
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


def metrics(y_test, y_hat, price_point):
    # Run Price Specific Measures
    lstm_rmse = rmse(y_test, y_hat)
    lstm_mape = mape(y_test, y_hat)
    lstm_smape = smape_original(y_test, y_hat)
    lstm_smape_adjusted = smape_adjusted(y_test, y_hat)

    # run Precision, Recall, Accuracy, and F-Measure on model's ability to predict
    # the next day is up or down

    y_true_up, y_predictions_up = custom_accuracy(y_test, y_hat)

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

    # return lstm_rmse, lstm_mape, lstm_smape, lstm_smape_adjusted, \
    #        precision, recall, accuracy, f_measure


def main():
    # PREPROCESS

    # Global Constants
    # set Batch Size:
    BATCH_SIZE = 100
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.001

    # SPX

    # get data from CSV:
    spx_data = pd.read_csv('../../data/spx_prices.csv')

    # reverse data:
    spx_data = spx_data.iloc[::-1]

    # reset the index of the dataframe:
    spx_open_price = spx_data.reset_index()['Open']
    spx_high_price = spx_data.reset_index()['High']
    spx_low_price = spx_data.reset_index()['Low']
    spx_close_price = spx_data.reset_index()['Close']

    # normalize data:
    scaler = MinMaxScaler(feature_range=(0, 1))

    # fit the data with the scaler:
    spx_fit_open = scaler.fit_transform(np.array(spx_open_price).reshape(-1, 1))
    spx_fit_high = scaler.fit_transform(np.array(spx_high_price).reshape(-1, 1))
    spx_fit_low = scaler.fit_transform(np.array(spx_low_price).reshape(-1, 1))
    spx_fit_close = scaler.fit_transform(np.array(spx_close_price).reshape(-1, 1))

    training_size = int(len(spx_data) * 0.65)

    # spx train data
    train_spx_fit_open = spx_fit_open[0: training_size]
    spx_open_x_train, spx_open_y_train = batch_data(train_spx_fit_open, BATCH_SIZE)
    train_spx_fit_high = spx_fit_high[0: training_size]
    spx_high_x_train, spx_high_y_train = batch_data(train_spx_fit_high, BATCH_SIZE)
    train_spx_fit_low = spx_fit_low[0: training_size]
    spx_low_x_train, spx_low_y_train = batch_data(train_spx_fit_low, BATCH_SIZE)
    train_spx_fit_close = spx_fit_close[0: training_size]
    spx_close_x_train, spx_close_y_train = batch_data(train_spx_fit_close, BATCH_SIZE)

    spx_x_train = tf.concat([spx_open_x_train, spx_high_x_train, spx_low_x_train, spx_close_x_train], axis=1)
    # reshape for model:
    spx_x_train = tf.reshape(spx_x_train, (spx_x_train.shape[0], spx_x_train.shape[1], 1))
    spx_y_train = tf.concat([spx_open_y_train, spx_high_y_train, spx_low_y_train, spx_close_y_train], axis=1)

    # spx test data
    test_spx_fit_open = spx_fit_open[training_size: len(spx_data)]
    spx_open_x_test, spx_open_y_test = batch_data(test_spx_fit_open, BATCH_SIZE)
    test_spx_fit_high = spx_fit_high[training_size: len(spx_data)]
    spx_high_x_test, spx_high_y_test = batch_data(test_spx_fit_high, BATCH_SIZE)
    test_spx_fit_low = spx_fit_low[training_size: len(spx_data)]
    spx_low_x_test, spx_low_y_test = batch_data(test_spx_fit_low, BATCH_SIZE)
    test_spx_fit_close = spx_fit_close[training_size: len(spx_data)]
    spx_close_x_test, spx_close_y_test = batch_data(test_spx_fit_close, BATCH_SIZE)

    spx_x_test = tf.concat([spx_open_x_test, spx_high_x_test, spx_low_x_test, spx_close_x_test], axis=1)
    # reshape for model:
    spx_x_test = tf.reshape(spx_x_test, (spx_x_test.shape[0], spx_x_test.shape[1], 1))
    spx_y_test = tf.concat([spx_open_y_test, spx_high_y_test, spx_low_y_test, spx_close_y_test], axis=1)
    scaled_spx_y_test = scaler.inverse_transform(spx_y_test)

    # Run LSTM Model:
    lstm, predicted_stock_price = create_lstm_model(spx_x_train, spx_y_train, spx_x_test,
                                                    BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS)

    predicted_open = predicted_stock_price[:, 0]
    predicted_high = predicted_stock_price[:, 1]
    predicted_low = predicted_stock_price[:, 2]
    predicted_close = predicted_stock_price[:, 3]

    # Open:
    real_open_price = scaler.inverse_transform(spx_open_y_test)
    predicted_open = predicted_open.reshape(-1, 1)
    scaled_lstm_predicted_open = scaler.inverse_transform(predicted_open)
    metrics(spx_open_y_test, predicted_open, "Open")
    visualize_results(real_open_price, scaled_lstm_predicted_open, "TEST PREDICTIONS for Open (CNN-LSTM Model)", 0)

    # High:
    real_high_price = scaler.inverse_transform(spx_high_y_test)
    predicted_high = predicted_high.reshape(-1, 1)
    scaled_lstm_predicted_high = scaler.inverse_transform(predicted_high)
    metrics(spx_high_y_test, predicted_high, "High")
    visualize_results(real_high_price, scaled_lstm_predicted_high, "TEST PREDICTIONS for High (CNN-LSTM Model)", 1)

    # Low:
    real_low_price = scaler.inverse_transform(spx_low_y_test)
    predicted_low = predicted_low.reshape(-1, 1)
    scaled_lstm_predicted_low = scaler.inverse_transform(predicted_low)
    metrics(spx_low_y_test, predicted_low, "Low")
    visualize_results(real_low_price, scaled_lstm_predicted_low, "TEST PREDICTIONS for Low (CNN-LSTM Model)", 2)

    # Close:
    real_close_price = scaler.inverse_transform(spx_close_y_test)
    predicted_close = predicted_close.reshape(-1, 1)
    scaled_lstm_predicted_close = scaler.inverse_transform(predicted_close)
    metrics(spx_close_y_test, predicted_close, "Close")
    visualize_results(real_close_price, scaled_lstm_predicted_close, "TEST PREDICTIONS for Close (CNN-LSTM Model)", 3)

    # get dates:
    # Date:
    dates = spx_data.reset_index()["Date"]
    test_dates = dates[training_size: len(dates)]
    dates_for_charts = test_dates[BATCH_SIZE: len(test_dates)].reset_index()["Date"]

    # build dataframe for graphing:
    df = pd.DataFrame(columns=["Open", "High", "Low", "Close"])
    df["Open"] = np.reshape(scaled_lstm_predicted_open, newshape=(len(predicted_open, )))
    df["High"] = np.reshape(scaled_lstm_predicted_high, newshape=(len(predicted_high, )))
    df["Low"] = np.reshape(scaled_lstm_predicted_low, newshape=(len(predicted_low, )))
    df["Close"] = np.reshape(scaled_lstm_predicted_close, newshape=(len(predicted_close, )))

    title2 = "SPX Predicted Candles using CNN-LSTM Model on Test"

    create_candlestick_graph(dates_for_charts, df["Open"], df["High"], df["Low"], df["Close"], title2)


if __name__ == '__main__':
    main()
