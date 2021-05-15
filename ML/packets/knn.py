import load_data
import tree_extractor

import numpy as np
import sklearn
import sklearn.neighbors
import sklearn.metrics


def port_preprocess(port):
	if port > 1023:
		return 1024

	return port

X_train, X_test, y_train, y_test = load_data.get()

# feature pre-processing

X_train[:, 2] = np.array(list(map(port_preprocess, X_train[:, 2])))
X_test[:, 2] = np.array(list(map(port_preprocess, X_test[:, 2])))

X_train[:, 3] = np.array(list(map(port_preprocess, X_train[:, 3])))
X_test[:, 3] = np.array(list(map(port_preprocess, X_test[:, 3])))

X_train[:, 4] = np.array(list(map(lambda x: int(x!=0), X_train[:, 4])))
X_test[:, 4] = np.array(list(map(lambda x: int(x!=0), X_test[:, 4])))

# remove delays, TCP PSH
X_train = np.delete(X_train, obj=(5,6), axis=1)
X_test = np.delete(X_test, obj=(5,6), axis=1)


X_train_old, y_train_old = X_train, y_train

# training test = 1/2, validation set = 1/4, testing set = 1/4
X_train, X_val, y_train, y_val =\
	   sklearn.model_selection.train_test_split(X_train, y_train, test_size=0.3333333333333, random_state = 42)

best_acc = -1
best_conf = 0
best_k = -1

print("MODEL SELECTION ---------")

# hyperparameter optimization (n_neighbors)
for k in range(1, 11):

	classifier = sklearn.neighbors.KNeighborsClassifier(n_neighbors=k)
	classifier = classifier.fit(X_train, y_train)

	y_pred = classifier.predict(X_val)

	confusion_matrix = sklearn.metrics.confusion_matrix(y_val, y_pred)
	tn, fp, fn, tp = confusion_matrix.ravel()
	accuracy = (tp + tn) / y_val.shape[0]

	if accuracy > best_acc:
		best_k = k
		best_acc = accuracy
		best_conf = confusion_matrix

print(best_conf)
print(best_acc, best_k)

print("MODEL EVALUATION ---------")

# evaluate model performance

X_train, y_train = X_train_old, y_train_old

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

classifier = sklearn.neighbors.KNeighborsClassifier(n_neighbors=best_k)
classifier.fit(X_train, y_train)

tree_extractor.extract_tree_evaluation(classifier, X_train,\
	                                   X_test, y_train, y_test)