import load_data
import numpy as np
import sklearn
import sklearn.model_selection
import sklearn.tree
import sklearn.metrics

import tree_extractor
from sklearn.tree import export_text

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

print("MODEL SELECTION ---------")

best_acc = -1
best_conf = 0
best_depth = -1

# hyperparameter optimization (tree depth)
for i in range(3, 50):
	classifier = sklearn.tree.DecisionTreeClassifier(max_depth=i, random_state=42)
	classifier = classifier.fit(X_train, y_train)

	y_pred = classifier.predict(X_val)

	confusion_matrix = sklearn.metrics.confusion_matrix(y_val, y_pred)
	tn, fp, fn, tp = confusion_matrix.ravel()
	accuracy = (tp + tn) / y_val.shape[0]

	print(accuracy, i)

	if accuracy > best_acc:
		best_acc = accuracy
		best_conf = confusion_matrix
		best_depth = i

print(best_conf)
print(best_acc, best_depth)

print("MODEL EVALUATION ---------")

# evaluate model performance

X_train, y_train = X_train_old, y_train_old

classifier = sklearn.tree.DecisionTreeClassifier(random_state=42, max_depth=best_depth)
classifier = classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

confusion_matrix = sklearn.metrics.confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix.ravel()
accuracy = (tp + tn) / y_test.shape[0]

print(confusion_matrix)
print(accuracy)

print("--- FINAL TREE ---")

# retrain final model using entire dataset
X = np.concatenate((X_train, X_test), axis=0)
y = np.concatenate((y_train, y_test))

classifier = sklearn.tree.DecisionTreeClassifier(random_state=42, max_depth=best_depth)
classifier = classifier.fit(X, y)

y_pred = classifier.predict(X)

confusion_matrix = sklearn.metrics.confusion_matrix(y, y_pred)
tn, fp, fn, tp = confusion_matrix.ravel()
accuracy = (tp + tn) / y.shape[0]


print(confusion_matrix)
print(accuracy)

print("--- RULE EXTRACTION --- ")

# extract rules from the decision tree
tree_extractor.extract_tree_rules(classifier, X, ["proto","dst","dport","sport","chksum","payload_len"])
