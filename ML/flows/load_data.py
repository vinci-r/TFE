import numpy as np
import sklearn
import sklearn.model_selection

def get():
	
	goodX = np.genfromtxt('good_features2.csv', delimiter=',')
	goodY = np.zeros(goodX.shape[0])
	badX = np.genfromtxt('bad_features2.csv', delimiter=',')
	badY = np.ones(badX.shape[0])

	X = np.concatenate((goodX, badX), axis=0)
	y = np.concatenate((goodY, badY))

	X_train, X_test, y_train, y_test =\
	   sklearn.model_selection.train_test_split(X, y, test_size=0.25, random_state = 42)

	return X_train, X_test, y_train, y_test