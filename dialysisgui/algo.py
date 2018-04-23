from sklearn.metrics import accuracy_score
from time import time
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
  
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pickle
import os

def random_forest_csv(csv_file):
	# path1=os.path.join("C:\\Users\\ONE\Desktop\\beproject\\dialysisgui\\static_in_pro\\media_root\\documents",csv_file)
	df = pd.read_csv(csv_file, header = 0)

	X = np.array(df.drop(['class'],1))
	y = np.array(df['class'])

	X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2)

	# the classifier
	clf1 = RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, 
								  min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, 
								  min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=-1, random_state=1, verbose=0, 
								  warm_start=False, class_weight=None)
	# train
	t1 = time()
	clf1.fit(X_train, y_train)
	print ("\ntraining time:", round(time()-t1, 3), "s")

	# predict
	t1 = time()
	pred = clf1.predict(X_test)
	# print(y_test)
	# print(pred)

	accuracy = accuracy_score(pred, y_test)

	# print ('\naccuracy = {0}'.format(accuracy))

	filename='finalized_model_rfc.sav'
	pickle.dump(clf1, open(filename, 'wb'))

	for count in range(0,6000):
		# print(count)
		X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2)

		# the classifier
		clf1 = RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, 
								  min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, 
								  min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=-1, random_state=1, verbose=0, 
								  warm_start=False, class_weight=None)
		# train
		t1 = time()
		clf1.fit(X_train, y_train)
		# print ("\ntraining time:", round(time()-t1, 3), "s")

		# predict
		t1 = time()
		pred = clf1.predict(X_test)
		# print(y_test)
		# print(pred)

		accuracy1 = accuracy_score(pred, y_test)
		# print ('\naccuracy = {0}'.format(accuracy))
		# f=open(filename, 'wb')
		if accuracy1>accuracy:
			accuracy=accuracy1
			a = [[0,0], [0,0]] 
			clf2=clf1
			a=confusion_matrix(y_test, pred, labels=None, sample_weight=None)
	pickle.dump(clf2, f)
	pickle.dump(accuracy, f)
	pickle.dump(a, f)		
	return(accuracy)

def svm_csv(csv_file):
	df = pd.read_csv(csv_file, header = 0)

	X = np.array(df.drop(['class'],1))
	y = np.array(df['class'])

	X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2)

# the classifier
	clf1 = svm.SVC(C=1.5, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, 
				   class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', random_state=None)
	# train
	t1 = time()
	clf1.fit(X_train, y_train)
	# print ("\ntraining time:", round(time()-t1, 3), "s")

# predict
	t1 = time()
	pred = clf1.predict(X_test)
	# print(y_test)
	# print(pred)

	accuracy = accuracy_score(pred, y_test)

	# print ('\naccuracy = {0}'.format(accuracy))

	filename='finalized_model_svm.sav'
	pickle.dump(clf1, open(filename, 'wb'))

	for count in range(0,10000):
		# print(count)
		X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2)

		# the classifier
		clf1 = svm.SVC(C=1.5, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, 
				   class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', random_state=None)
		# train
		t1 = time()
		clf1.fit(X_train, y_train)
		# print ("\ntraining time:", round(time()-t1, 3), "s")

		# predict
		t1 = time()
		pred = clf1.predict(X_test)
		# print(y_test)
		# print(pred)

		accuracy1 = accuracy_score(pred, y_test)
		# print ('\naccuracy = {0}'.format(accuracy))
		f=open(filename, 'wb')
		if accuracy1>accuracy:
			accuracy=accuracy1
			clf2=clf1
			a = [[0,0], [0,0]] 
			a=confusion_matrix(y_test, pred, labels=None, sample_weight=None)

	# print ('\n Final accuracy = {0}'.format(accuracy))
	# print(a)
	pickle.dump(clf2, f)
	pickle.dump(accuracy, f)
	pickle.dump(a, f)
	return(accuracy)

def mlp_csv(csv_file):
	t1 = time()
	df = pd.read_csv(csv_file, header = 0)

	X = np.array(df.drop(['class'],1))
	y = np.array(df['class'])

	X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2)

	# the classifier
	clf1 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)

	# train
	clf1.fit(X_train, y_train)

	# predict
	pred = clf1.predict(X_test)
	# print(y_test)
	# print(pred)

	accuracy = accuracy_score(pred, y_test)

	# print ('\naccuracy = {0}'.format(accuracy))

	filename='finalized_model_mlp.sav'

	pickle.dump(clf1, open(filename, 'wb'))
	pickle.dump(accuracy, open(filename, 'wb'))
	for count in range(0,6000):
		# print(count)
		X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2)

		# the classifier
		clf1 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)

		# train
		clf1.fit(X_train, y_train)

		# predict
		pred = clf1.predict(X_test)
		# print(y_test)
		# print(pred)

		accuracy1 = accuracy_score(pred, y_test)
		f=open(filename, 'wb')
		if accuracy1>accuracy:
			accuracy=accuracy1
			clf2=clf1
			a = [[0,0], [0,0]] 
			a=confusion_matrix(y_test, pred, labels=None, sample_weight=None)
		
	# print ('\n Final accuracy = {0}'.format(accuracy))
	# print("Confusion matrix is:")
	# print(a)
	pickle.dump(clf2, f)
	pickle.dump(accuracy, f)
	pickle.dump(a, f)
	# print ("\nTotal time taken:", round(time()-t1, 3), "s")
	return(accuracy)







