import numpy as np
import sklearn
import sklearn.model_selection

def get():
	
	goodX = np.genfromtxt('good_features.csv', delimiter=',')
	goodY = np.zeros(goodX.shape[0])
	badX = np.genfromtxt('bad_features.csv', delimiter=',')
	badY = np.ones(badX.shape[0])

	X = np.concatenate((goodX, badX), axis=0)
	y = np.concatenate((goodY, badY))

# ______ ONLY KEEP REQUESTS _______ 

	newX = []
	newy = []

	k = 0
	for i in X:
		if i[1] == 3232235523:
			newX.append(i)
			newy.append(y[k])
		k += 1

	X = np.array(newX)
	y = np.array(newy) 

# ________________________________ 

	X_train, X_test, y_train, y_test =\
	   sklearn.model_selection.train_test_split(X, y, test_size=0.25, random_state = 42)

	return X_train, X_test, y_train, y_test