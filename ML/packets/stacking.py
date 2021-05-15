import load_data
import tree_extractor

import numpy as np
import sklearn
import sklearn.ensemble
import sklearn.metrics


X_train, X_test, y_train, y_test = load_data.get()

def port_preprocess(port):
	if port > 1023:
		return 1024

	return port

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


# hyperparameters values determined empirically when considering the individual models
estimators = [
    ('rf', sklearn.ensemble.RandomForestClassifier(max_depth=14, random_state=42)),
    ('knn', sklearn.neighbors.KNeighborsClassifier(n_neighbors=4))
]
classifier = sklearn.ensemble.StackingClassifier(
	                 stack_method='predict',
                     estimators=estimators,
	                 final_estimator=sklearn.tree.DecisionTreeClassifier(max_depth=10, random_state=42))

print("MODEL EVALUATION ---------")

# evaluate model performance

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

confusion_matrix = sklearn.metrics.confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix.ravel()
accuracy = (tp + tn) / y_test.shape[0]

print(confusion_matrix)
print(accuracy)


print("TREE EXTRACTION EVALUATION ---------")

# extract a tree and evaluate performance

tree_extractor.extract_tree_evaluation(classifier, X_train,\
	                                   X_test, y_train, y_test)