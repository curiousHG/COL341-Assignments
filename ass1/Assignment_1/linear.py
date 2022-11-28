import numpy as np
from numpy.linalg.linalg import norm
import scipy as sp
import sys
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoLars
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

def moore_penrose_pseudoinverse(X,y,lamb):
    return np.linalg.inv(np.dot(X.T,X) + lamb*np.identity(X.shape[1])).dot(X.T).dot(y)
def normerror(y,x,w):
    return np.linalg.norm(y-np.dot(x,w))/np.linalg.norm(y)
def simple_linear_reg():
    train_data = sys.argv[2]
    test_data = sys.argv[3]
    data = pd.read_csv(train_data)
    data_y = data["Total Costs"]
    data_x = data.drop("Total Costs", axis=1)
    data_x = data_x.iloc[:,1:]
    X = np.array(data_x)
    X = np.c_[np.ones(X.shape[0]),X]
    y = np.array(data_y)
    w = moore_penrose_pseudoinverse(X,y,0)
    test_Data = pd.read_csv(test_data)
    test_Data = test_Data.iloc[:,1:]
    test_Data = np.c_[np.ones(test_Data.shape[0]),test_Data]
    output = np.dot(test_Data,w)
    np.savetxt(f'{sys.argv[4]}', output, delimiter='\n')
    np.savetxt(f'{sys.argv[5]}', w, delimiter='\n')

def train_kFold_with_r(r,data,folds):
    acc = np.zeros(folds)
    for i in range(folds):
        train_data = np.array(data[data.fold != i])
        test_data = np.array(data[data.fold == i])
        x = train_data[:,1:-2]
        x = np.c_[np.ones(x.shape[0]),x]
        y = train_data[:,-2]
        test_x = test_data[:,1:-2]
        test_x = np.c_[np.ones(test_x.shape[0]),test_x]
        test_y = test_data[:,-2]
        w = moore_penrose_pseudoinverse(x,y,r)
        acc[i] = normerror(test_y,test_x,w)
    return np.mean(acc)

def ridge_regression():
    train_data = sys.argv[2]
    test_data = sys.argv[3]
    reg_file = sys.argv[4]
    data = pd.read_csv(train_data)
    reg = np.loadtxt(reg_file,delimiter = '\n')
    folds = 10
    data["fold"] = data.index.values%folds
    d = {'regularization':[], 'accuracy':[]}
    for r in reg:
        d['regularization'].append(r)
        v = train_kFold_with_r(r,data,folds)
        d['accuracy'].append(v)
    mini = np.argmin(d['accuracy'])
    mini_r = d['regularization'][mini]
    fdata = np.array(data.drop(["fold"],axis = 1))
    x = fdata[:,1:-1]
    x = np.c_[np.ones(x.shape[0]),x]
    y = fdata[:,-1]
    w = moore_penrose_pseudoinverse(x,y,mini_r)
    test = np.array(pd.read_csv(test_data))
    test_x = test[:,1:]
    test_x = np.c_[np.ones(test_x.shape[0]),test_x]
    output = np.dot(test_x,w)
    np.savetxt(sys.argv[5],output)
    np.savetxt(sys.argv[6],w)
    np.savetxt(sys.argv[7],np.array([mini_r]))

def reduce_feature():
    data = sys.argv[2]
    predict_data = sys.argv[3]
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

    useful = [
        213, 78, 204, 128, 87, 303, 211, 165, 99, 205, 48, 33, 321, 263, 182, 210, 217, 
        226, 212, 91, 46, 308, 215, 221, 282, 207, 74, 218, 238, 288, 186, 296, 77, 112, 
        56, 85, 24, 287, 52, 247, 75, 125, 86, 119, 307, 315, 216, 140, 81, 173, 304, 164, 
        279, 163, 265, 267, 101, 50, 84, 95, 300, 273, 102, 42, 295, 80, 60, 299
    ]
    active_X = np.c_[np.ones(X_poly.shape[0]),X_poly[:,useful]]

    #CROSS VALIDATION
    lambdas = [0.001,0.003,0.01,0.03,0.1,0.3,1,3,10,30,100,300,1000]
    # train_x,test_x,train_y,test_y = train_test_split(active_X,y,test_size=0.7,random_state=42)

    # print("Start cv")
    # fold = np.random.randint(0,10,size = train_x.shape[0])
    # d = {'regularization':[], 'accuracy':[]}
    # def cv(r,cv_x,cv_y,folds):
    #     acc = np.zeros(folds)
    #     for i in range(folds):
    #         train_cv_x = cv_x[np.where(fold != i)]
    #         test_cv_x = cv_x[np.where(fold == i)]
    #         train_cv_y = cv_y[fold != i]
    #         test_cv_y = cv_y[fold == i]
    #         w = moore_penrose_pseudoinverse(train_cv_x,train_cv_y,r)
    #         acc[i] = r2_score(test_cv_y,np.dot(test_cv_x,w))
    #     return np.mean(acc)

    # for r in lambdas:
    #     d['regularization'].append(r)
    #     v = cv(r,train_x,train_y,10)
    #     d['accuracy'].append(v)
    #     print(f"{r} {v}")
    # print("Done CV")
    # mini = np.argmax(d['accuracy'])
    # mini_r = d['regularization'][mini]
    # print(mini_r)
    mini_r = 10
    w = moore_penrose_pseudoinverse(active_X,y,mini_r)
    pred_data = pd.read_csv(predict_data)
    pred_data.drop(to_drop,axis=1,inplace=True)
    poly_pred = poly.fit_transform(pred_data)
    pred_X = np.c_[np.ones(poly_pred.shape[0]),poly_pred[:,useful]]
    pred_y = np.dot(pred_X,w)
    np.savetxt(sys.argv[4],pred_y)

if sys.argv[1] == 'a':
    simple_linear_reg()
elif sys.argv[1] == 'b':
    ridge_regression()
elif sys.argv[1] == 'c':
    reduce_feature()