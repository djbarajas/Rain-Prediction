from xgboost import XGBClassifier
from sklearn import metrics
import pickle
import numpy as np


class Level:

    # for initializing train and test sets, classifier and accuracy score
    def __init__(self):
        self.X_train = []
        self.X_labels = []
        self.test = []
        self.test_labels = []
        self.model = XGBClassifier(learning_rate=0.01)
        self.prediction = 0
        self.error = 0

    # adding the data points
    # the sin and cos are to compute the sin and cos of 0-364 which are days of the year.
    # This adds more useful, easily obtained features.
    def input_train_data(self, year, month, day, hour, sin, cos, latitude, longitude, feature, rainfall):
        data_point = [year, month, day, hour, sin, cos, latitude, longitude, feature]
        self.X_train.append(data_point)
        self.X_labels.append(rainfall)

    # train the data
    def train(self):
        self.X_train = np.asarray(self.X_train)
        self.X_labels = np.asarray(self.X_labels)
        self.model.fit(self.X_train, self.X_labels)

    # input test data
    def input_test_data(self, year, month, day, hour, sin, cos, latitude, longitude, feature):
        data_point = [year, month, day, hour, sin, cos, latitude, longitude, feature]
        self.test.append(data_point)

    # input test labels if you want to check accuracy
    def label(self, label):
        self.test_labels.append(label)

    # test data
    def predict(self):
        self.test = np.asarray(self.test)
        self.prediction = self.model.predict(self.test)
        return self.prediction

    # if you have the test labels you can check the error rate (you want error close to 0)
    def check_error(self):
        self.test_labels = np.asarray(self.test_labels)
        self.error = metrics.mean_absolute_error(self.test_labels, self.prediction)
        return self.error

    # save classifier
    def save_classifier(self, classifier_name):
        filename = classifier_name + ".pkl"
        pickle.dump(self.model, open(filename, 'wb'))

    # open saved classifier
    def open_classifier(self, classifier_name):
        filename = classifier_name + ".pkl"
        self.model = pickle.load(open(filename, 'rb'))

    # removes all training data
    def clean_train(self):
        self.X_train = []
        self.X_labels = []

    # removes all testing data
    def clean_train(self):
        self.test = []
        self.test_labels = []