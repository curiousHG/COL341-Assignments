import numpy as np
import pandas as pd


train_path = 'data/train.csv'
test_path = 'data/test.csv'

train = pd.read_csv(train_path, index_col = 0)    
test = pd.read_csv(test_path, index_col = 0)
    
y_train = np.array(train['Length of Stay'])

train = train.drop(columns = ['Length of Stay'])

#Ensuring consistency of One-Hot Encoding

data = pd.concat([train, test], ignore_index = True)
cols = train.columns
cols = cols[:-1]
data = pd.get_dummies(data, columns=cols, drop_first=True)
data = data.to_numpy()
X_train = data[:train.shape[0], :]
X_test = data[train.shape[0]:, :]
