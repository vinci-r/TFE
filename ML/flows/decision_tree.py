import load_data
import numpy as np
import sklearn
import sklearn.model_selection
from sklearn.model_selection import KFold
import sklearn.tree
import sklearn.metrics

from sklearn.tree import export_text

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


print("MODEL SELECTION ---------")

best_acc = -1
best_conf = 0
best_depth = -1

kf = KFold(n_splits=10)

for i in range(3, 45):

	accuracies = []

	for indices in kf.split(X_train):

		X_train1 = np.array([X_train[j] for j in indices[0]])
		y_train1 = np.array([y_train[j] for j in indices[0]])

		X_val = np.array([X_train[j] for j in indices[1]])
		y_val = np.array([y_train[j] for j in indices[1]])

		classifier = sklearn.tree.DecisionTreeClassifier(max_depth=i, random_state=42)
		classifier = classifier.fit(X_train1, y_train1)

		y_pred = classifier.predict(X_val)

		confusion_matrix = sklearn.metrics.confusion_matrix(y_val, y_pred)
		tn, fp, fn, tp = confusion_matrix.ravel()
		accuracy = (tp + tn) / y_val.shape[0]
		accuracies.append(accuracy)

	accuracy = np.average(accuracies)
	if accuracy > best_acc:
		best_acc = accuracy
		best_conf = confusion_matrix
		best_depth = i


print(best_conf)
print(best_acc, best_depth)

print("MODEL EVALUATION ---------")

# evaluate model performance

classifier = sklearn.tree.DecisionTreeClassifier(random_state=42, max_depth=best_depth)
classifier = classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

confusion_matrix = sklearn.metrics.confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix.ravel()
accuracy = (tp + tn) / y_test.shape[0]

print(confusion_matrix)
print(accuracy)