import load_data
import tree_extractor

from sklearn.model_selection import KFold
import numpy as np
import sklearn
import sklearn.neighbors
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

X_train[:, 3] = np.array(list(map(lambda x: int(x!=0), X_train[:, 3])))
X_test[:, 3] = np.array(list(map(lambda x: int(x!=0), X_test[:, 3])))

# remove average delays, TCP PSH
X_train = np.delete(X_train, obj=(4,6), axis=1)
X_test = np.delete(X_test, obj=(4,6), axis=1)


best_acc = -1
best_conf = 0
best_k = -1

print("MODEL SELECTION ---------")

kf = KFold(n_splits=10)

# hyperparameter optimization (n_neighbors)
for k in range(1, 11):

	accuracies = []

	for indices in kf.split(X_train):

		X_train1 = np.array([X_train[j] for j in indices[0]])
		y_train1 = np.array([y_train[j] for j in indices[0]])

		X_val = np.array([X_train[j] for j in indices[1]])
		y_val = np.array([y_train[j] for j in indices[1]])

		classifier = sklearn.neighbors.KNeighborsClassifier(n_neighbors=k)
		classifier = classifier.fit(X_train, y_train)

		y_pred = classifier.predict(X_val)

		confusion_matrix = sklearn.metrics.confusion_matrix(y_val, y_pred)
		tn, fp, fn, tp = confusion_matrix.ravel()
		accuracy = (tp + tn) / y_val.shape[0]
		accuracies.append(accuracy)

	accuracy = np.average(accuracies)

	if accuracy > best_acc:
		best_k = k
		best_acc = accuracy
		best_conf = confusion_matrix

print(best_conf)
print(best_acc, best_k)

print("MODEL EVALUATION ---------")

# evaluate model performance

classifier = sklearn.neighbors.KNeighborsClassifier(n_neighbors=best_k)
classifier = classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

confusion_matrix = sklearn.metrics.confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix.ravel()
accuracy = (tp + tn) / y_test.shape[0]

print(confusion_matrix)
print(accuracy)


print("TREE EXTRACTION EVALUATION ---------")

# extract a tree and evaluate performance

classifier = sklearn.neighbors.KNeighborsClassifier(n_neighbors=11)
classifier.fit(X_train, y_train)

tree_extractor.extract_tree_evaluation(classifier, X_train,\
	                                   X_test, y_train, y_test)