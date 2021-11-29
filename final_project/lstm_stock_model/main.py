import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from stock_model import StockModel
import tensorflow as tf
import matplotlib.pyplot as plt
from random import randrange

def batch_data(data, batch_size=1):
    """
    batches data

    :param data: data to batch
    :param batch_size: size of the batch
    :return:
    """

    dataX, dataY = [], []

    overall_length = len(data)

    for i in range(0, overall_length - (overall_length % batch_size)):
        x = data[i: i + batch_size]
        y = data[i + 1: i + 1 + batch_size]

        if len(y) == batch_size and len(x) == batch_size:
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


def test(model, x_test, y_test):
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

        # add price prediction (as actual prices) to output list
        correct_predictions.append(test_predictions[-1][0])

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

    # get our data by train and test:
    train_data = fit_data[0: training_size]
    test_data = fit_data[training_size: len(fit_data)]

    # BEGIN MODEL

    # init our model:
    model = StockModel(batch_size=100)

    # batch the train data:
    x_train, y_train = batch_data(train_data, model.batch_size)
    train_data_without_scale = \
        scaler.inverse_transform(np.array(train_data[model.batch_size: len(train_data)]).reshape(-1, 1))

    # batch the test data:
    x_test, y_test = batch_data(test_data, model.batch_size)
    test_data_without_scale = \
        scaler.inverse_transform(np.array(test_data[model.batch_size: len(test_data)]).reshape(-1, 1))

    # train the model:
    NUM_EPOCHS = 1

    for epoch in range(NUM_EPOCHS):
        train(model, x_train, y_train)
        model.reset_states()

    # run model on train
    train_precision, train_recall, train_accuracy, train_f_measure, train_predictions = test(model, x_train, y_train)
    train_predictions_without_scale = scaler.inverse_transform(train_predictions)
    visualize_results(train_data_without_scale, train_predictions_without_scale, "TRAIN DATA", 0)
    print("###MODEL TRAIN RESULTS###")
    print("-------------------")
    print("Average TRAIN Precision: " + str(train_precision))
    print("-------------------")
    print("Average TRAIN Recall: " + str(train_recall))
    print("-------------------")
    print("Average TRAIN Accuracy: " + str(train_accuracy))
    print("-------------------")
    print("Average TRAIN F-Measure: " + str(train_f_measure))
    print("-------------------")
    print("###END OF RESULTS###")

    print("--------------------")

    # run model on test
    precision, recall, accuracy, f_measure, predictions = test(model, x_test, y_test)

    predictions = np.array(predictions.copy()).reshape(-1, 1)
    predictions_without_scale = scaler.inverse_transform(predictions)

    print("###MODEL TEST RESULTS###")
    print("-------------------")
    print("Average Test Precision: " + str(precision))
    print("-------------------")
    print("Average Test Recall: " + str(recall))
    print("-------------------")
    print("Average Test Accuracy: " + str(accuracy))
    print("-------------------")
    print("Average Test F-Measure: " + str(f_measure))
    print("-------------------")
    print("###END OF RESULTS###")

    visualize_results(test_data_without_scale, predictions_without_scale, "TEST DATA", 1)


if __name__ == '__main__':
    main()
