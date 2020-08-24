from xgboost import XGBRegressor
from sklearn import metrics
import pickle
import numpy as np


class Level:

    # for initializing train and test sets, classifier and accuracy score
    # Change method to gpu_hist if you want xgboost to run on a GPU
    def __init__(self, method='auto'):
        self.X_train = []
        self.X_labels = []
        self.test = []
        self.test_labels = []
        self.model = XGBRegressor(tree_method=method, objective='reg:squarederror', learning_rate=0.01)
        self.prediction = 0
        self.error = 0

    # adding the data points
    def input_train_data(self, features, rainfall):
        if isinstance(self.X_train, np.ndarray) and self.X_train.size > 0:
            self.X_train = self.X_train.tolist()
            self.X_labels = self.X_labels.tolist()
        self.X_train.append(features)
        self.X_labels.append(rainfall)

    # adding the data points
    # the sin and cos are to compute the sin and cos of 0-364 which are days of the year.
    # This adds more useful, easily obtained features.
    def input_train_data1(self, year, month, day, hour, sin, cos, latitude, longitude, feature, rainfall):
        if isinstance(self.X_train, np.ndarray) and self.X_train.size > 0:
            self.X_train = self.X_train.tolist()
            self.X_labels = self.X_labels.tolist()
        data_point = [year, month, day, hour, sin, cos, latitude, longitude, feature]
        self.X_train.append(data_point)
        self.X_labels.append(rainfall)

    # train the data
    def train(self):
        self.X_train = np.asarray(self.X_train)
        self.X_labels = np.asarray(self.X_labels)
        self.model.fit(self.X_train, self.X_labels)

        # self.X_train = []
        # self.X_labels = []

    # input test data
    def input_test_data2(self, year, month, day, hour, sin, cos, latitude, longitude, feature):
        if isinstance(self.test, np.ndarray) and self.test.size > 0:
            self.test = self.test.tolist()
        data_point = [year, month, day, hour, sin, cos, latitude, longitude, feature]
        self.test.append(data_point)

    def input_test(self, features):
        if isinstance(self.test, np.ndarray) and self.test.size > 0:
            self.test = self.test.tolist()
        self.test.append(features)

    # input test labels if you want to check accuracy
    def label(self, label):
        self.test_labels.append(label)

    # test data
    def predict(self):
        self.test = np.asarray(self.test)
        self.prediction = self.model.predict(self.test)
        # self.test = []
        # self.test_labels = []
        return self.prediction

    # if you have the test labels you can check the error rate (you want error close to 0)
    def check_error(self):
        self.test_labels = np.asarray(self.test_labels)
        self.error = metrics.mean_absolute_error(self.test_labels, self.prediction)
        return self.error

    # save classifier
    def save_classifier(self, classifierName):
        filename = classifierName + ".pkl"
        pickle.dump(self.model, open(filename, 'wb'))

    # open saved classifier
    def open_classifier(self, classifierName):
        filename = classifierName + ".pkl"
        self.model = pickle.load(open(filename, 'rb'))

    # removes all training data
    def clean_train(self):
        self.X_train = []
        self.X_labels = []

    # removes all testing data
    def clean_test(self):
        self.test = []
        self.test_labels = []