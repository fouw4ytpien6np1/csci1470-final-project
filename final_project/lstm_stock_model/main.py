import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from stock_model import StockModel
import tensorflow as tf
import matplotlib.pyplot as plt


def batch_data(data, batch_size=1):
    """
    batches data

    :param data: data to batch
    :param batch_size: size of the batch
    :return:
    """

    dataX, dataY = [], []

    overall_length = len(data)

    for i in range(0, overall_length):
        if overall_length - i < batch_size:
            continue
        x = data[i: i + batch_size]
        y = data[i + 1: i + 1 + batch_size]

        if len(x) == 100 and len(y) == 100:
            dataX.append(x)
            dataY.append(y)

    return np.array(dataX), np.array(dataY)


def train(model, x_train, y_train):
    """
    trains the model for 1 epoch

    :param model: model to train
    :param x_train: the x training data
    :param y_train: the y training data
    :param scaler: the scaler used to normalize the data
    :return: None

    """
    for i in range(len(x_train)):
        if i == 0:
            continue
        with tf.GradientTape() as tape:
            predictions = model.call(x_train[i])
            predictions = tf.reshape(predictions, shape=(model.batch_size, 1))
            loss = model.loss(y_train[i], predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def test(model, x_test, y_test, scaler):
    """
    tests the trained model

    :param model: trained model to test
    :param x_test: the x test data
    :param y_test: the y test data
    :param scaler: the scaler uses to normalize the data
    :return: avg_precision, avg_recall, avg_accuracy, avg_f_measure, correct_test_predictions
    """
    correct_predictions = []
    precisions = []
    recalls = []
    accuracies = []
    f_measures = []

    for i in range(len(x_test)):

        # make predictions from trained model:
        test_predictions = model.call(x_test[i])

        test_predictions = tf.reshape(test_predictions, shape=(model.batch_size, 1))

        # add price prediction (as actual prices) to output list
        correct_price_prediction = scaler.inverse_transform(test_predictions)
        correct_predictions.append(correct_price_prediction[-1][0] / 2)

        # calculate precision:
        precision_metric = tf.keras.metrics.Precision()
        precision_metric.update_state(test_predictions, y_test[i])
        precision = precision_metric.result().numpy()
        precisions.append(precision)

        # calculate recall:
        recall_metric = tf.keras.metrics.Recall()
        recall_metric.update_state(test_predictions, y_test[i])
        recall = recall_metric.result().numpy()
        recalls.append(recall)

        # calculate accuracy
        accuracy_metric = tf.keras.metrics.Accuracy()
        accuracy_metric.update_state(test_predictions, y_test[i])
        accuracy = accuracy_metric.result().numpy()
        accuracies.append(accuracy)

        # calculate f_measure
        f_measure = 2 * precision * recall / (precision + recall)
        f_measures.append(f_measure)

    return np.sum(precisions) / len(precisions), np.sum(recalls) / len(recalls), \
           np.sum(accuracies) / len(accuracies), np.sum(f_measures) / len(accuracies), correct_predictions


def visualize_results(real_stock_price, predicted_stock_price):
    plt.plot(real_stock_price, color='black', label='Stock Price')
    plt.plot(predicted_stock_price, color='green', label='Predicted Stock Price')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()


def main():
    # PREPROCESS

    # get data from CSV:
    data = pd.read_csv('../../data/spx_prices.csv')

    # reverse data
    data = data.iloc[::-1]

    # reset the index of the dataframe:
    reset_data = data.reset_index()[' Close']

    # LSTM sensitive to data scale, so normalize:
    scaler = MinMaxScaler(feature_range=(0, 1))

    # fit the data with the scaler:
    fit_data = scaler.fit_transform(np.array(reset_data).reshape(-1, 1))

    # get our training size (65% of the data):
    training_size = int(len(fit_data) * 0.65)

    # test size, which is the difference between the data and the training size:
    test_size = len(fit_data) - training_size

    # get our data by train and test:
    train_data, test_data = fit_data[0:training_size, :], fit_data[test_size:len(fit_data), :1]
    test_data_without_scale = scaler.inverse_transform(test_data)

    # init our model:
    model = StockModel(batch_size=100)

    # batch the train data:
    x_train, y_train = batch_data(train_data, model.batch_size)

    # reshape x_train to 3D tensor for LSTM
    x_train = tf.reshape(x_train, shape=(x_train.shape[0], x_train.shape[1], 1))

    # batch the test data:
    x_test, y_test = batch_data(test_data, model.batch_size)

    # reshape x_test to 3D tensor for LSTM
    x_test = tf.reshape(x_test, shape=(x_test.shape[0], x_test.shape[1], 1))

    # train the model:

    NUM_EPOCHS = 5

    for epoch in range(NUM_EPOCHS):
        train(model, x_train, y_test)
        model.reset_states()

    precision, recall, accuracy, f_measure, predictions = test(model, x_test, y_test, scaler)

    print("###MODEL RESULTS###")
    print("-------------------")
    print("Average Precision: " + str(precision))
    print("-------------------")
    print("Average Recall: " + str(recall))
    print("-------------------")
    print("Average Accuracy: " + str(accuracy))
    print("-------------------")
    print("Average F-Measure: " + str(f_measure))
    print("-------------------")
    print("###END OF RESULTS###")

    visualize_results(test_data_without_scale, predictions)


if __name__ == '__main__':
    main()
