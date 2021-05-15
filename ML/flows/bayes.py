import load_data
import tree_extractor

import numpy as np
import sklearn
import sklearn.naive_bayes
import sklearn.metrics

def power_of_10(x):
	if x == 0:
		return 0
	if x > 0 and x < 1e-7:
		return 1
	if x >= 1e-7 and x < 1e-6:
		return 2
	if x >= 1e-6 and x < 1e-5:
		return 3
	if x >= 1e-5 and x < 1e-4:
		return 4
	if x >= 1e-4 and x < 1e-3:
		return 5
	if x >= 1e-3 and x < 1e-2:
		return 6
	if x >= 1e-2 and x < 1e-1:
		return 7
	if x >= 1e-1 and x < 1.0:
		return 8
	if x >= 1.0 and x < 1e1:
		return 9
	if x >= 1e1 and x < 1e2:
		return 10
	if x >= 1e2 and x < 1e3:
		return 11
	if x >= 1e3 and x < 1e4:
		return 12
	if x >= 1e4 and x < 1e5:
		return 13
	
	return 14

def port_preprocess(port):
	if port > 1023:
		return 1024

	return port

X_train, X_test, y_train, y_test = load_data.get()


# feature pre-processing

X_train[:, 1] = np.array(list(map(port_preprocess, X_train[:, 1])))
X_test[:, 1] = np.array(list(map(port_preprocess, X_test[:, 1])))

X_train[:, 2] = np.array(list(map(port_preprocess, X_train[:, 2])))
X_test[:, 2] = np.array(list(map(port_preprocess, X_test[:, 2])))

# this will perform poorly, data is not discrete, non obvious discretization

X_train[:, 3] = np.array(list(map(lambda x: int(x!=0), X_train[:, 3])))
X_test[:, 3] = np.array(list(map(lambda x: int(x!=0), X_test[:, 3])))

X_train[:, 4] = np.array(list(map(power_of_10, X_train[:, 4])))
X_test[:, 4] = np.array(list(map(power_of_10, X_test[:, 4])))

X_train[:, 5] = np.array(list(map(power_of_10, X_train[:, 5])))
X_test[:, 5] = np.array(list(map(power_of_10, X_test[:, 5])))

X_train[:, 6] = np.array(list(map(power_of_10, X_train[:, 6])))
X_test[:, 6] = np.array(list(map(power_of_10, X_test[:, 6])))

# remove delays, TCP PSH
X_train = np.delete(X_train, obj=(2,4,6), axis=1)
X_test = np.delete(X_test, obj=(2,4,6), axis=1)


# alpha parameter does not have any influence
best_a = 1.0

print("MODEL EVALUATION ---------")

# evaluate model performance

classifier = sklearn.naive_bayes.MultinomialNB(alpha=best_a)
classifier = classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

confusion_matrix = sklearn.metrics.confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix.ravel()
accuracy = (tp + tn) / y_test.shape[0]

print(confusion_matrix)
print(accuracy)


print("TREE EXTRACTION EVALUATION ---------")

# extract a tree and evaluate performance

classifier = sklearn.naive_bayes.MultinomialNB(alpha=best_a)
classifier.fit(X_train, y_train)

tree_extractor.extract_tree_evaluation(classifier, X_train,\
	                                   X_test, y_train, y_test)