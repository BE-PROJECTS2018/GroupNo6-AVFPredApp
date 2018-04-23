# Load scikit's random forest classifier library
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

# Load pandas
import pandas as pd

# Load numpy
import numpy as np

import pickle
import os
#v1=
#v2=
def mlp(v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16,v17,v18,v19,v20,v21,v22,v23,v24,v25,v26,v27,v28,v29):
	
	clf1=pickle.load(open('finalized_model_mlp.sav', 'rb'))
	temp=[v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16,v17,v18,v19,v20,v21,v22,v23,v24,v25,v26,v27,v28,v29]
	temp = np.array(temp).reshape((1, -1))
	pred = clf1.predict(temp)
	if pred==0:
		pred = 'Fistula health seems fine. Continue with dialysis.'
	else:
		pred = 'Fistula seems to have deteriorated. Ultrasonic Doppler required for confirmation. Continue with high flux dialyser until test results are back.'
	return pred
    