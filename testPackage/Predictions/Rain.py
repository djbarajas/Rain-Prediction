
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle


class Rain:

	def __init__(self):
		self.X_train = []
		self.X_labels = []
		self.test = []
		self.test_labels = []
		self.rf_clf = RandomForestClassifier(n_estimators=10)
		self.prediction = 0
		self. acc = 0

	# adding the data points
	def input_train_data(self, features, rainfall):
		self.X_train.append(features)
		self.X_labels.append(rainfall)

	#adding the data points
	#the sin and cos are to compute the sin and cos of 0-364 which are days of the year.  This adds kore useful easily obtained features.
	def input_train_data1(self, year, month, day, hour, sin, cos, latitude, longitude, feature, rainfall):
		dataPoint = [year, month, day, hour, sin, cos, latitude, longitude, feature]
		self.X_train.append(dataPoint)
		self.X_labels.append(rainfall)

	#train the data
	def train(self):
		self.rf_clf.fit(self.X_train,self.X_labels)
		self.X_train = []
		self.X_labels = []

	#input test data
	def input_test_data(self, year, month, day, hour, sin, cos, latitude, longitude, feature):
		dataPoint = [year, month, day, hour, sin, cos, latitude, longitude, feature]
		self.test.append(dataPoint)

	def input_test(self, features):

		if len(self.test) > 0 and isinstance(self.test, np.ndarray):
			self.test = self.test.tolist()
		self.test.append(features)

	#input test labels
	def label(self, label):
		self.test_labels.append(label)

	#test data
	def predict(self):
		self.prediction = self.rf_clf.predict(self.test)
		self.test = []
		self.test_labels = []
		return self.prediction

	#check probability
	def predict_prob(self):
		probability = self.rf_clf.predict_proba(self.test)
		self.test = []
		self.test_labels = []
		return probability

	# check accuracy
	def accuracy_score(self):
		if(len(self.prediction) == len(self.test_labels)):
			self.acc = accuracy_score(self.prediction,self.test_labels)
			print(self.acc)
		else:
			print("unable to check accuracy score until all data is labeled")



	def useful_features(self):
		# copied from:  https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
		import matplotlib.pyplot as plt
		importances = self.rf_clf.feature_importances_
		std = np.std([tree.feature_importances_ for tree in self.rf_clf.estimators_],axis=0)

		indices = np.argsort(importances)[::-1]

		# Print the feature ranking
		print("Feature ranking:")

		for f in range(len(self.X_train[1])):
		    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

		# Plot the feature importances of the forest
		plt.figure()
		plt.title("Feature importances")
		plt.bar(range(len(self.X_train[1])), importances[indices],
		       color="r", yerr=std[indices], align="center")
		plt.xticks(range(len(self.X_train[1])), indices)
		plt.xlim([-1, len(self.X_train[1])])
		plt.show()


	#save classifier
	def save_classifier(self, classifierName):
		filename = classifierName + ".pkl"
		pickle.dump(self.rf_clf, open(filename, 'wb'))


	#open saved classifier
	def open_classifier(self, classifierName):
		filename = classifierName + ".pkl"
		self.rf_clf = pickle.load(open(filename, 'rb'))

	# removes all training data
	def clean_train(self):
		self.X_train = []
		self.X_labels = []

	# removes all testing data
	def clean_train(self):
		self.test = []
		self.test_labels = []

