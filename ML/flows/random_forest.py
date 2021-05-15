import load_data
import tree_extractor

from sklearn.model_selection import KFold
import numpy as np
import sklearn
import sklearn.ensemble
import sklearn.metrics


X_train, X_test, y_train, y_test = load_data.get()


best_n_estimators = 0
best_depth = 0
best_conf = 0
max_acc = -1

def port_preprocess(port):
	if port > 1023:
		return 1024

	return port

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

print("MODEL SELECTION ---------")

kf = KFold(n_splits=10)

# hyperparameter optimization (max_depth)
for i in range(1, 25):

	accuracies = []

	for indices in kf.split(X_train):

		X_train1 = np.array([X_train[j] for j in indices[0]])
		y_train1 = np.array([y_train[j] for j in indices[0]])

		X_val = np.array([X_train[j] for j in indices[1]])
		y_val = np.array([y_train[j] for j in indices[1]])

		classifier = sklearn.ensemble.RandomForestClassifier(max_depth=i, random_state=42)
		classifier = classifier.fit(X_train1, y_train1)

		y_pred = classifier.predict(X_val)

		confusion_matrix = sklearn.metrics.confusion_matrix(y_val, y_pred)
		tn, fp, fn, tp = confusion_matrix.ravel()
		accuracy = (tp + tn) / y_val.shape[0]
		accuracies.append(accuracy)

	accuracy = np.average(accuracies)
	print(accuracy, i)

	if accuracy > max_acc:
		max_acc = accuracy
		best_conf = confusion_matrix
		best_depth = i


print(best_conf)
print(max_acc, best_depth)

print("MODEL EVALUATION ---------")

# evaluate model performance

classifier = sklearn.ensemble.RandomForestClassifier(max_depth=best_depth, random_state=42)
classifier = classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

confusion_matrix = sklearn.metrics.confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix.ravel()
accuracy = (tp + tn) / y_test.shape[0]

print(confusion_matrix)
print(accuracy)

print("TREE EXTRACTION EVALUATION ---------")

# extract a tree and evaluate performance

classifier = sklearn.ensemble.RandomForestClassifier(max_depth=best_depth, random_state=42)
classifier.fit(X_train, y_train)

tree_extractor.extract_tree_evaluation(classifier, X_train,\
	                                   X_test, y_train, y_test)