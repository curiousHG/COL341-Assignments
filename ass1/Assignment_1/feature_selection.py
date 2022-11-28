import numpy as np
from numpy.linalg.linalg import norm
import scipy as sp
import sys
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoLars

data = "data/train_large.csv"
# predict_data = sys.argv[3]
data = pd.read_csv(data)
to_drop = [  
    "Facility Id",
    "CCS Procedure Code",
    "CCS Diagnosis Code",
    "APR DRG Code",
    "APR MDC Code",
    "APR Severity of Illness Code",
    "Unnamed: 0"
]
data.drop(to_drop,axis=1,inplace=True)
X = data.drop(['Total Costs'],axis = 1)
y = data['Total Costs']

poly = PolynomialFeatures(degree=2,include_bias=False)
X_poly = poly.fit_transform(X)

sampling_set = np.random.choice(X.shape[0],size=int(X.shape[0]*0.8),replace=False)
lars_X = X_poly[sampling_set]
lars_y = y[sampling_set]
model = LassoLars(alpha=0.1).fit(lars_X,lars_y)

# active_X = np.c_[np.ones(X_poly.shape[0]),X_poly[:,model.active_]]
print(model.active_)