import numpy as np
import sklearn
import sklearn.model_selection
import sklearn.tree
import sklearn.metrics

# extract a tree and evaluate its performance
def extract_tree_evaluation(classifier, X_train, X_test, y_train, y_test):
	
	y_pred = classifier.predict(X_train)

	X_new = []
	y_new = []

	# the final tree is built based on well classified examples
	for i in range(y_train.shape[0]):
		if y_pred[i] == y_train[i]:
			X_new.append(X_train[i])
			y_new.append(y_train[i])

	X_new = np.array(X_new)
	y_new = np.array(y_new)


	X_train_old, y_train_old = X_train, y_train

	X_train, X_val, y_train, y_val =\
		   sklearn.model_selection.train_test_split(X_train, y_train, test_size=0.3333333333333, random_state = 42)

	best_acc = -1
	best_conf = 0
	best_depth = -1

	# hyperparameter optimization (tree depth)
	for i in range(10, 70):
		tree_classifier = sklearn.tree.DecisionTreeClassifier(max_depth=i, random_state=42)
		tree_classifier = tree_classifier.fit(X_train, y_train)

		y_pred = tree_classifier.predict(X_val)

		confusion_matrix = sklearn.metrics.confusion_matrix(y_val, y_pred)
		tn, fp, fn, tp = confusion_matrix.ravel()
		accuracy = (tp + tn) / y_val.shape[0]

		if accuracy > best_acc:
			best_acc = accuracy
			best_conf = confusion_matrix
			best_depth = i

	# evaluate model performance

	X_train, y_train = X_train_old, y_train_old

	tree_classifier = sklearn.tree.DecisionTreeClassifier(random_state=42, max_depth=best_depth)
	tree_classifier = tree_classifier.fit(X_train, y_train)


	y_pred = tree_classifier.predict(X_test)

	confusion_matrix = sklearn.metrics.confusion_matrix(y_test, y_pred)
	tn, fp, fn, tp = confusion_matrix.ravel()
	accuracy = (tp + tn) / y_test.shape[0]

	print(best_depth, accuracy)