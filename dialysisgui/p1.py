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

def mlp_csv(csv_file):
    
    clf1=pickle.load(open('finalized_model_mlp.sav', 'rb'))
    path1=os.path.join("static_in_pro\\media_root\\documents",csv_file)

    df_file = pd.read_csv(path1, header = 0)

    X_file = np.array(df_file)

    # predict
    pred_file = clf1.predict(X_file)
    print(pred_file)
    return pred_file



