import pandas as pd
import numpy as np
import sys


class Activation_function():
    def __init__(self, activation_function):
        self.name = activation_function
        self.function = self.activation_function()
        self.derivative = self.activation_function_derivative()

    def activation_function(self):
        if self.name == 'sigmoid':
            return lambda x: 1 / (1 + np.exp(-x))
        elif self.name == 'relu':
            return lambda x: np.maximum(0, x)
        elif self.name == 'tanh':
            return lambda x: np.tanh(x)
        elif self.name == 'softmax':
            return lambda x: np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)),axis = 0)

    def activation_function_derivative(self):
        if self.name == 'sigmoid':
            return lambda x: x * (1 - x)
        elif self.name == 'relu':
            return lambda x: 1. * (x > 0)
        elif self.name == 'tanh':
            return lambda x: 1 - np.power(x, 2)
        elif self.name == 'softmax':
            return lambda x: x*(1-x)


def loss_function_CE(y_true, y_pred, derivative):
    if not derivative:
        return -np.sum(y_true * np.log(y_pred))/y_true.shape[1]
    else:
        return (y_pred-y_true)/y_true.shape[1]


def loss_function_MSE(y_true, y_pred, derivative=False):
    if not derivative:
        return np.sum((y_true - y_pred) ** 2) / 2
    else:
        temp = np.sum(2*(y_pred - y_true)*y_pred,axis = 0)
        f = 2*(y_pred - y_true)*y_pred*(1-y_pred) - y_pred*temp
        return f/y_true.shape[1]


class Neural_Network():

    def __init__(
        self, input_size, hidden_layer_size_array, output_size,
        activation_function, output_activation_function, loss_function
    ):
        self.input_size = input_size
        self.number_of_hidden_layers = len(hidden_layer_size_array)
        self.hidden_layer_size_array = hidden_layer_size_array
        self.output_size = output_size
        self.activation_function = Activation_function(activation_function)
        self.output_activation_function = Activation_function(
            output_activation_function)
        self.weights = self.weights_initializer()
        self.loss_function = loss_function
        self.weights_history = []

    def weights_initializer(self):
        weights = []
        for i in range(self.number_of_hidden_layers+1):
            if i == 0:
                weights.append(
                    np.random.normal(
                        size=(self.input_size + 1, self.hidden_layer_size_array[i]))
                    * np.sqrt(2 / (self.input_size + self.hidden_layer_size_array[i] + 1))
                    .astype(np.float32)
                )
            elif i == self.number_of_hidden_layers:
                weights.append(
                    np.random.normal(
                        size=(self.hidden_layer_size_array[i - 1] + 1, self.output_size))
                    * np.sqrt(2 / (self.hidden_layer_size_array[i - 1] + self.output_size + 1))
                    .astype(np.float32)
                )
            else:
                weights.append(
                    np.random.normal(
                        size=(self.hidden_layer_size_array[i - 1] + 1, self.hidden_layer_size_array[i]))
                    * np.sqrt(2 / (self.hidden_layer_size_array[i - 1] + self.hidden_layer_size_array[i] + 1))
                    .astype(np.float32)
                )
        return weights

    def feed_forward(self, input_data):
        # Feed forward
        # Input layer
        input_layer = np.array(input_data, ndmin=2).T

        # Hidden layers
        a_s = []
        z_s = []
        for i in range(self.number_of_hidden_layers):
            input_layer = np.concatenate(
                (np.ones((1, input_layer.shape[1])), input_layer), axis=0)
            z = np.dot(self.weights[i].T, input_layer)
            z_s.append(z)
            a_s_i = self.activation_function.function(z)
            a_s.append(a_s_i)
            input_layer = a_s[i]
            a_s[i] = np.concatenate(
                (np.ones((1, a_s[i].shape[1])), a_s[i]), axis=0)

        # Output layer
        input_layer = np.concatenate(
            (np.ones((1, input_layer.shape[1])), input_layer), axis=0)
        z = np.dot(self.weights[-1].T, input_layer)
        z_s.append(z)
        a_s.append(self.output_activation_function.function(z))
        return a_s[-1], a_s, z_s

    def back_propagation(self, input_data, y_true):
        input_data = np.array(input_data, ndmin=2)
        y_pred, a_s, z_s = self.feed_forward(input_data)

        # Output layer
        y_true = np.array(y_true, ndmin=2).T

        # delL_dzs = [(y_pred-y_true)/y_true.shape[1]]
        delL_dzs = [self.loss_function(y_true, y_pred, derivative=True)]
        delL_das = [np.dot(self.weights[-1], delL_dzs[-1])]

        # Hidden layers
        for i in range(self.number_of_hidden_layers-1, -1, -1):
            delL_dz_i = delL_das[-1] * \
                self.activation_function.derivative(a_s[i])
            delL_dzs.append(delL_dz_i[1:])
            delL_das.append(np.dot(self.weights[i], delL_dzs[-1]))

        delL_dzs.reverse()
        delL_dws = []
        for i in range(self.number_of_hidden_layers+1):
            if i == 0:
                input_data = np.c_[np.ones(input_data.shape[0]), input_data]
                delL_dws.append(np.dot(input_data.T, delL_dzs[i].T))
            else:
                delL_dws.append(np.dot(a_s[i-1], delL_dzs[i].T))

        return delL_dws

    def train(self, input_data, y_true, epochs, batch_size, learning_rate, adaptive_learning_rate=False):
        # Training
        iter = 0
        for i in range(epochs):
            for j in range(0, len(input_data), batch_size):
                if j + batch_size > len(input_data):
                    batch_input_data = input_data[j:]
                    batch_y_true = y_true[j:]
                else:
                    batch_input_data = input_data[j:j+batch_size]
                    batch_y_true = y_true[j:j+batch_size]
                # del_w = self.back_propagation(batch_input_data, batch_y_true)
                del_w = self.back_propagation(batch_input_data, batch_y_true)
                if adaptive_learning_rate == True:
                    learning_rate = learning_rate * np.sqrt(1/(iter+1))
                for k in range(len(self.weights)):
                    self.weights[k] -= learning_rate * del_w[k]
                iter += 1

    def predict(self, input_data):
        input_data = np.array(input_data, ndmin=2)
        y_pred, _, _ = self.feed_forward(input_data)
        return y_pred

    def evaluate(self, input_data, y_true):
        y_pred = self.predict(input_data)
        return self.loss_function(y_true.T, y_pred,False)


input_path = sys.argv[1]
output_path = sys.argv[2]
if input_path[-1] != '/':
    input_path += '/'
if output_path[-1] != '/':
    output_path += '/'

train_file = f'{input_path}toy_dataset_train.csv'
test_file = f'{input_path}toy_dataset_test.csv'
param_file = sys.argv[3]

with open(param_file, 'r') as f:
    param = f.readlines()

epochs = int(param[0])
batch_size = int(param[1])

param[2] = param[2][1:-2]
hidden_Layer = list(map(int, param[2].split(',')))
out_size = hidden_Layer[-1]
hidden_Layer = hidden_Layer[:-1]

adaptive = bool(int(param[3]))
lr = float(param[4])

ac = int(param[5])
if ac == 0:
    activation_function = 'sigmoid'
elif ac == 1:
    activation_function = 'tanh'
elif ac == 2:
    activation_function = 'relu'

l_f = int(param[6])
if l_f == 0:
    loss_function = loss_function_CE
elif l_f == 1:
    loss_function = loss_function_MSE

seed = int(param[7])
np.random.seed(seed)

train_data = pd.read_csv(train_file, header=None)
X_train = train_data.iloc[:, 1:].values
X_train = X_train.astype(np.float)
X_train = X_train / 255
y_train = train_data.iloc[:, 0].values
y_true = np.array(pd.get_dummies(y_train))

nn = Neural_Network(input_size=X_train.shape[1],
                    hidden_layer_size_array=hidden_Layer,
                    output_size=out_size,
                    activation_function=activation_function,
                    output_activation_function='softmax',
                    loss_function=loss_function,
                    )

nn.train(
    X_train, y_true,
    epochs=epochs, batch_size=batch_size,
    learning_rate=lr,
    adaptive_learning_rate=adaptive
)

for i in range(len(nn.weights)):
    np.save(f'{output_path}/w_{i+1}.npy', nn.weights[i])

X_test = pd.read_csv(test_file, header=None)
X_test = X_test.iloc[:, 1:].values

preds = nn.predict(X_test)
preds = np.argmax(preds, axis=0)

np.save(f'{output_path}/predictions.npy', preds)