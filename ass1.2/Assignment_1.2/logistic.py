import sys
import numpy as np
import scipy as sp
from scipy.special import softmax
import pandas as pd
import time
from sklearn.feature_selection import SelectKBest, f_classif

def load_data():
    train_path = sys.argv[2]
    test_path = sys.argv[3]
    paramfile = sys.argv[4]
    outputfile = sys.argv[5]
    weightfile = sys.argv[6]

    train = pd.read_csv(train_path, index_col = 0)    
    test = pd.read_csv(test_path, index_col = 0)
    y_train = np.array(train['Length of Stay'])
    train = train.drop(columns = ['Length of Stay'])
    data = pd.concat([train, test], ignore_index = True)
    cols = train.columns
    cols = cols[:-1]
    data = pd.get_dummies(data, columns=cols, drop_first=True)
    data = data.to_numpy()
    X_train = data[:train.shape[0], :]
    X_test = data[train.shape[0]:, :]
    b = np.ones((X_train.shape[0], 1))
    X_train = np.concatenate((b,X_train), axis=1)
    b = np.ones((X_test.shape[0], 1))
    X_test = np.concatenate((b,X_test), axis=1)
    y_true = pd.get_dummies(y_train)

    y_true = y_true.to_numpy()
    w = np.zeros(shape = (X_train.shape[1], y_true.shape[1]))
    return X_train, X_test, y_true, w, paramfile, outputfile, weightfile

def load_data_2():
    train_path = sys.argv[2]
    test_path = sys.argv[3]
    outputfile = sys.argv[4]
    weightfile = sys.argv[5]

    train = pd.read_csv(train_path, index_col = 0)    
    test = pd.read_csv(test_path, index_col = 0)
    y_train = np.array(train['Length of Stay'])
    train = train.drop(columns = ['Length of Stay'])
    data = pd.concat([train, test], ignore_index = True)
    cols = train.columns
    cols = cols[:-1]
    data = pd.get_dummies(data, columns=cols, drop_first=True)
    data = data.to_numpy()
    X_train = data[:train.shape[0], :]
    X_test = data[train.shape[0]:, :]
    b = np.ones((X_train.shape[0], 1))
    X_train = np.concatenate((b,X_train), axis=1)
    b = np.ones((X_test.shape[0], 1))
    X_test = np.concatenate((b,X_test), axis=1)
    y_true = pd.get_dummies(y_train)
    y_true = y_true.to_numpy()
    
    w = np.zeros(shape = (X_train.shape[1], y_true.shape[1]))
    return X_train, X_test, y_true, w, outputfile, weightfile

def load_data_3():
    train_path = sys.argv[2]
    test_path = sys.argv[3]
    outputfile = sys.argv[4]
    weightfile = sys.argv[5]

    train = pd.read_csv(train_path, index_col = 0)    
    test = pd.read_csv(test_path, index_col = 0)
    y_train = np.array(train['Length of Stay'])
    train = train.drop(columns = ['Length of Stay'])
    data = pd.concat([train, test], ignore_index = True)
    cols = train.columns
    cols = cols[:-1]
    data = pd.get_dummies(data, columns=cols, drop_first=True)
    data = data.to_numpy()
    X_train = data[:train.shape[0], :]
    X_test = data[train.shape[0]:, :]
    b = np.ones((X_train.shape[0], 1))
    X_train = np.concatenate((b,X_train), axis=1)
    b = np.ones((X_test.shape[0], 1))
    X_test = np.concatenate((b,X_test), axis=1)    
    return X_train, X_test, y_train, outputfile, weightfile

def y_hat(X,w):
    return softmax(np.dot(X,w),axis = 1)

def gradient(X,w,y_true):
    return np.dot(X.T,(y_hat(X,w)-y_true))/X.shape[0]

def loss(X,w,y_true):
    k =np.take_along_axis(y_hat(X,w), np.argmax(y_true,axis = 1)[:,None], axis=1)
    return -np.sum(np.log(k))/X.shape[0]

def fixed_gradient(X,y_true,w,n_iter,lr):
    step = 0
    while step<n_iter:
        w -= lr*gradient(X,w,y_true)
        step += 1
    return w

def adaptive_gradient(X,y_true,w,n_iter,lr):
    step = 0
    while step<n_iter:
        w -= (lr/np.sqrt(step+1))*gradient(X,w,y_true)
        step += 1
    return w

def ab_backtrack(X,y_true,w,alpha,beta,lr):
    new_lr = lr
    grad = gradient(X,w,y_true)
    diff = -alpha*new_lr*(np.linalg.norm(grad, ord = 'fro')**2)
    curr_loss = loss(X,w,y_true)
    new_loss = loss(X,w - new_lr*grad,y_true)

    while new_loss - curr_loss> diff:
        new_lr *= beta
        new_loss = loss(X,w - new_lr*grad,y_true)
        diff *= beta
    return new_lr

def alpha_beta_gradient(X,y_true,w,n_iter,alpha,beta,lr):
    step = 0
    while step<n_iter:
        new_lr = ab_backtrack(X,y_true,w,alpha,beta,lr)
        step+=1
        w -= new_lr*gradient(X,w,y_true)
        
    return w

def mini_batch_gradient(X,y_true,w,n_iter,batch_size,lr,params):
    steps = X.shape[0]//batch_size
    i= 0
    while i<n_iter:
        if params[0] == 1:
            for j in range(steps):
                w = fixed_gradient(X[j*batch_size:(j+1)*batch_size],y_true[j*batch_size:(j+1)*batch_size],w,1,lr)
        elif params[0] == 2:
            for j in range(steps):
                w = adaptive_gradient(X[j*batch_size:(j+1)*batch_size],y_true[j*batch_size:(j+1)*batch_size],w,1,lr)
        elif params[0] == 3:
            for j in range(steps):
                w = alpha_beta_gradient(X[j*batch_size:(j+1)*batch_size],y_true[j*batch_size:(j+1)*batch_size],w,1,params[1],params[2],lr)
        i+=1
    return w

def mini_batch_gradient_best(X,y_true,w,batch_size,lr,start_time,limit):
    
    steps = X.shape[0]//batch_size
    i = 0
    j = 0
    while j<steps and time.time()-start_time<limit:
        w = adaptive_gradient(X[j*batch_size:(j+1)*batch_size],y_true[j*batch_size:(j+1)*batch_size],w,1,lr)
        # alpha = 0.1
        # beta = 0.9
        # w = alpha_beta_gradient(X[j*batch_size:(j+1)*batch_size],y_true[j*batch_size:(j+1)*batch_size],w,1,alpha,beta,lr)
        i+=1
        j+=1
        if j==steps:
            j=0
    return w

if sys.argv[1]=='a':
    X_train, X_test, y_true, w, paramfile, outputfile, weightfile = load_data()
    with open(paramfile) as f:
        case = f.readlines()
    for i in range(len(case)):
        case[i] = case[i].strip()
    case[0] = int(case[0])
    if case[0] == 1:
        lr = float(case[1])
        n_iter = int(case[2])
        w_new = fixed_gradient(X_train,y_true,w,n_iter,lr)
    elif case[0] == 2:
        lr = float(case[1])
        n_iter = int(case[2])
        w_new = adaptive_gradient(X_train,y_true,w,n_iter,lr) 
    elif case[0] == 3:
        lr,alpha,beta = map(float,case[1].split(','))
        n_iter = int(case[2])
        w_new = alpha_beta_gradient(X_train,y_true,w,n_iter,alpha,beta,lr)
    y_test = y_hat(X_test,w_new)
    y_out = y_test.argmax(axis = 1)+1

    np.savetxt(outputfile, y_out, delimiter='\n')
    np.savetxt(weightfile, w_new.flatten(), delimiter='\n')

elif sys.argv[1]=='b':
    X_train, X_test, y_true, w, paramfile, outputfile, weightfile = load_data()
    with open(paramfile) as f:
        case = f.readlines()
    for i in range(len(case)):
        case[i] = case[i].strip()
    case[0] = int(case[0])
    if case[0] == 1:
        lr = float(case[1])
        n_iter = int(case[2])
        batch_size = int(case[3])
        w_new = mini_batch_gradient(X_train,y_true,w,n_iter,batch_size,lr,case)
    elif case[0] == 2:
        lr = float(case[1])
        n_iter = int(case[2])
        batch_size = int(case[3])
        w_new = mini_batch_gradient(X_train,y_true,w,n_iter,batch_size,lr,case)
    elif case[0] == 3:
        lr,alpha,beta = map(float,case[1].split(','))
        n_iter = int(case[2])
        batch_size = int(case[3])
        params = [case[0],alpha,beta]
        w_new = mini_batch_gradient(X_train,y_true,w,n_iter,batch_size,lr,params)
    y_test = y_hat(X_test,w_new)
    y_out = y_test.argmax(axis = 1)+1
    np.savetxt(outputfile, y_out, delimiter='\n')
    np.savetxt(weightfile, w_new.flatten(), delimiter='\n')
    
elif sys.argv[1]=='c':
    start_time = time.time()
    X_train,X_test,y_true,w,outputfile,weightfile = load_data_2()
    
    w_new = mini_batch_gradient_best(X_train,y_true,w,385,1.8774,start_time,limit=540)
    y_out = y_hat(X_test,w_new)
    # print("Loss Value",loss(X_train,w_new,y_true))
    np.savetxt(outputfile, y_out, delimiter='\n')
    np.savetxt(weightfile, w_new.flatten(), delimiter='\n')

elif sys.argv[1]=='d':
    start_time = time.time()
    X_train,X_test,y_true,w,outputfile,weightfile = load_data_2()
    w_new = mini_batch_gradient_best(X_train,y_true,w,385,1.8774,start_time,limit=840)
    y_out = y_hat(X_test,w_new)
    # print("Loss Value",loss(X_train,w_new,y_true))
    np.savetxt(outputfile, y_out, delimiter='\n')
    np.savetxt(weightfile, w_new.flatten(), delimiter='\n')