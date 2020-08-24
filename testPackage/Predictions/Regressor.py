from xgboost import XGBRegressor
from sklearn import metrics
import numpy as np
from sklearn.model_selection import train_test_split


class Regressor:

    # for initializing train and test sets, classifier and accuracy score
    # Change method to gpu_hist if you want xgboost to run on a GPU
    def __init__(self, params ={'objective':'reg:squarederror', 'verbosity':0}):
        self.X_train = []
        self.X_labels = []
        self.test = []
        self.test_labels = []
        self.model = XGBRegressor(**params)
        self.prediction = 0
        self.error = 0

    def size(self):
        if isinstance(self.X_train, np.ndarray):
            return self.X_train.size
        return len(self.X_train)

    # adding the data points
    def input_train(self, features, feature):
        if isinstance(self.X_train, np.ndarray) and self.X_train.size > 0:
            self.X_train = self.X_train.tolist()
            self.X_labels = self.X_labels.tolist()
        self.X_train.append(features)
        self.X_labels.append(feature)

    # train the data
    def train(self):
        self.X_train = np.asarray(self.X_train)
        self.X_labels = np.asarray(self.X_labels)
        self.model.fit(self.X_train, self.X_labels)

    def train_eval(self, metric='error'):
        self.X_train = np.asarray(self.X_train)
        self.X_labels = np.asarray(self.X_labels)
        X_train, X_test, y_train, y_test = train_test_split(self.X_train, self.X_labels, test_size=0.33)
        self.model.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            eval_metric=metric)
        evals_result = self.model.evals_result()
        if metric == 'error':
            validations = []
            for val in evals_result.values():
                lst = val.get("error")
                validations.append(sum(lst) / len(lst))
            return 1 - (sum(validations) / len(validations))
        else:
            validations = []
            for val in evals_result.values():
                lst = val.get(metric)
                validations.append(lst[-1])
            return validations

    # input test labels if you want to check accuracy
    def label(self, label):
        self.test_labels.append(label)

    def input_test(self, features):
        if isinstance(self.test, np.ndarray) and self.test.size > 0:
            self.test = self.test.tolist()
        self.test.append(features)

    # test data
    def predict(self):
        if not isinstance(self.test, np.ndarray):
            self.test = np.asarray(self.test)
        self.prediction = self.model.predict(self.test)
        return self.prediction

    # if you have the test labels you can check the error rate (you want error close to 0)
    def check_error(self):
        self.test_labels = np.asarray(self.test_labels)
        self.error = metrics.mean_absolute_error(self.test_labels, self.prediction)
        return self.error

    # save classifier
    def save_classifier(self, file):
        self.model.save_model(file)

    # open saved classifier
    def open_classifier(self, file):
        self.model.load_model(file)

    # removes all training data
    def clean_train(self):
        self.X_train = []
        self.X_labels = []

    # removes all testing data
    def clean_test(self):
        self.test = []
        self.test_labels = []
