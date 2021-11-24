import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from stock_model import StockModel
import tensorflow as tf


def batch_data(data, batch_size=1):
    """
    batches data

    :param data: data to batch
    :param batch_size: size of the batch
    :return:
    """

    dataX, dataY = [], []

    for i in range(len(data) - batch_size - 1):
        a = data[i:(i + batch_size), 0]
        dataX.append(a)
        dataY.append(data[i + batch_size, 0])

    return np.array(dataX), np.array(dataY)


def train(model, x_train, y_train, scaler):
    """
    trains the model for 1 epoch

    :param model: model to train
    :param x_train: the x training data
    :param y_train: the y training data
    :param scaler: the scaler used to normalize the data
    :return: None

    """
    for i in range(len(x_train)):
        with tf.GradientTape() as tape:
            predictions = model.call(x_train[i])
            correct_predictions = scaler.inverse_transform(predictions)
            loss = model.loss(y_train[i], correct_predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def test(model, x_test, y_test, scaler):
    """
    tests the trained model

    :param model: trained model to test
    :param x_test: the x test data
    :param y_test: the y test data
    :param scaler: the scaler uses to normalize the data
    :return: precision, recall, accuracy, f_measure
    """

    # make predictions from trained model:
    test_predictions = model.call(x_test)
    correct_test_predictions = scaler.inverse_transform(test_predictions)

    # calculate precision:
    precision_metric = tf.keras.metrics.Precision()
    precision_metric.update_state(correct_test_predictions, y_test)
    precision = precision_metric.result().numpy()

    # calculate recall:
    recall_metric = tf.keras.metrics.Recall()
    recall_metric.update_state(correct_test_predictions, y_test)
    recall = recall_metric.result().numpy()

    # calculate accuracy
    accuracy_metric = tf.keras.metrics.Accuracy()
    accuracy_metric.update_state(correct_test_predictions, y_test)
    accuracy = accuracy_metric.result().numpy()

    # calculate f_measure
    f_measure = 2 * precision * recall / (precision + recall)

    return precision, recall, accuracy, f_measure


def main():

    # PREPROCESS

    # get data from CSV:
    data = pd.read_csv('../data/spx_prices.csv')

    # reset the index of the dataframe:
    reset_data = data.reset_index()['close']

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

    # init our model:
    model = StockModel(batch_size=100)

    # batch the train data:
    x_train, y_train = batch_data(train_data, model.batch_size)
    # reshape x_train to 3D tensor for LSTM
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)

    # batch the test data:
    x_test, y_test = batch_data(test_data, model.batch_size)
    # reshape x_test to 3D tensor for LSTM
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    # train the model:

    NUM_EPOCHS = 5

    for epoch in range(NUM_EPOCHS):
        train(model, x_train, x_test, scaler)

    precision, recall, accuracy, f_measure = test(model, x_test, y_test, scaler)

    print("###MODEL RESULTS###")
    print("-------------------")
    print("Precision: " + str(precision))
    print("-------------------")
    print("Recall: " + str(recall))
    print("-------------------")
    print("Accuracy: " + str(accuracy))
    print("-------------------")
    print("F-Measure: " + str(f_measure))
    print("-------------------")
    print("###END OF RESULTS###")


if __name__ == '__main__':
    main()
