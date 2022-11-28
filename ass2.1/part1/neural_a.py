import pandas as pd
import numpy as np
import math
import sys

input_path = sys.argv[1]   # '../data/toy_dataset_train.csv'
output_path = sys.argv[2]   # 'output/'
param_path =  sys.argv[3]    # 'param.txt'

param = open(param_path, 'r')
params = param.readlines()

epochs = int(params[0])
batch_size = int(params[1])

def prep(a):
    a = a[1:-2]
    return list(map(int, a.split(',')))

layer_widths = prep(params[2])
learning_type = int(params[3])
learning_rate = float(params[4])
activation_hidden = int(params[5])
# activation_output = activation_hidden
loss_type = int(params[6])
seed = int(params[7])

train = pd.read_csv(input_path, header=None)
np.random.seed(seed)

class Network:

    def __init__(self, input_width, layer_widths, activation_hidden, activation_output, loss_type, batch_size, learning_type, learning_rate):
        
        # user given parameters 
        self.input_width = input_width
        self.layer_widths = layer_widths
        self.activation_hidden = activation_hidden
        self.activation_output = activation_output
        self.loss_type = loss_type
        self.batch_size = batch_size
        self.learning_type = learning_type
        self.learning_rate = learning_rate
        self.output_array = [None]*len(layer_widths)
        self.iterations = 0

        # initializing weights 
        self.initialize_weights()


    def initialize_weights(self):

        # array storing all the weight matrices
        self.parameters = []
        np.random.seed(seed)
        w = np.random.normal(0, 1, size=(1 + self.input_width, self.layer_widths[0]))*math.sqrt(2/(self.input_width + self.layer_widths[0] + 1))
        self.parameters.append(w)
        for i in range(len(self.layer_widths)-1):
            width_a, width_b = self.layer_widths[i], self.layer_widths[i+1]
            w = np.random.normal(0, 1, size=(1 + width_a, width_b))*math.sqrt(2/(width_a + width_b + 1))
            self.parameters.append(w)

    def forward(self, x):

        # returns y_hat
        x = np.concatenate((np.ones((x.shape[0], 1), dtype=np.float32), x), axis=1)
        # print(x)
        for i in range(len(self.layer_widths)-1):
            input = x.dot(self.parameters[i])
            x = self.activate_in_hidden(np.concatenate((np.ones((input.shape[0], 1), dtype=np.float32), input), axis=1))
            self.output_array[i] = x
        self.output_array[-1] = self.activate_in_output(x.dot(self.parameters[-1]))
        return self.output_array[-1]
    
    def dC_dajl_f(self, y_hat, y):

        if self.loss_type == 0:
            return ((y_hat-y)/y_hat.shape[0]).T
        if self.loss_type == 1:
            y_pred = y_hat.T
            y_true = y.T
            temp = np.sum(2*(y_pred - y_true)*y_pred, axis = 0)
            f = 2*(y_pred - y_true)*y_pred*(1-y_pred)
            f -= y_pred*temp
            return f.T/y_true.shape[1]

    def backward(self, y_hat, y):

        dC_dajl = self.dC_dajl_f(y_hat, y)
        l = len(self.layer_widths)
        i = l-2
        while i>=0:
            dz_dw = self.output_array[i].T
            ajs = self.output_array[i+1]

            if i == l-2: 
                da_dz = np.multiply(ajs, 1-ajs).T
                grad = dz_dw.dot(dC_dajl.T)
            else: 
                da_dz = np.multiply(ajs, 1-ajs).T[1:]
                grad = dz_dw.dot((da_dz*dC_dajl[1:]).T)
            
            if self.learning_type == 1: 
                self.learning_rate /= np.sqrt(self.iterations+1)
            self.parameters[i+1] -= self.learning_rate*grad

            if i == l-2: dC_dajl = self.parameters[i+1].dot(dC_dajl*da_dz)  
            else: dC_dajl = self.parameters[i+1].dot(dC_dajl[1:]*da_dz)  
            i -= 1
    
    def loss(self, y_hat, y):

        if self.loss_type == 1:
            return np.linalg.norm(y_hat-y)**2/y_hat.shape[0]
        if self.loss_type == 0:
            return -np.sum(np.sum(y*np.log(y_hat)))/y_hat.shape[0]

    def activate_in_hidden(self, x_w):
        
        if self.activation_hidden == 0:
            return 1/(1 + np.exp(-x_w))
        if self.activation_hidden == 1:
            return np.tanh(x_w)
        if self.activation_hidden == 2:
            return np.maximum(0, x_w)

    def activate_in_output(self, x_w):   

        return (np.exp(x_w - np.max(x_w)).T/np.sum(np.exp(x_w- np.max(x_w)), axis=1)).T


train_y = pd.get_dummies(train.iloc[:, 0])
train_x = np.array(train.iloc[:, 1:]/255)

input_width = train_x.shape[1]
activation_output = 'softmax'

network = Network(input_width, layer_widths, activation_hidden, activation_output, loss_type, batch_size, learning_type, learning_rate)
network.iterations = 0

epochs = 5
for epoch in range(epochs):
    for batch in range(train_x.shape[0]//batch_size):
        x = train_x[batch*batch_size:(batch+1)*batch_size]
        y = train_y[batch*batch_size:(batch+1)*batch_size]
        y_hat = network.forward(x) 
        # if network.iterations%100 == 0: print('Loss:', network.iterations, network.loss(y_hat, y))
        network.backward(y_hat, y)
        network.iterations += 1

w = network.parameters
for i in range(len(w)):
    k = np.save(output_path + '/w_' + str(i+1) + '.npy', w[i])
np.save(output_path + '/predictions.npy', network.forward(train_x))
