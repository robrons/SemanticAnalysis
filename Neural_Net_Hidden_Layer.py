# importing all the required libraries

import pandas as pd
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import csv

# My_X_Data.csv and My_Y_Data.csv imports the training data and result.  

x_train = np.array(pd.read_csv("My_X_Data.csv", header=None)) 

y_train = pd.read_csv("My_Y_Data.csv", header=None) 


# My_X_Test.csv and My_Y_Test.csv imports the testing data and result.  

x_val = np.array(pd.read_csv("My_X_Test_Data.csv", header=None)) 

y_val = pd.read_csv("My_Y_Test_Data.csv", header=None) 


# My_Weight_Train.csv imports the Bayesian weights.

weight = pd.read_csv("My_Weight_Train.csv", header=None)


# This step prints the dimensions of all the data that we imported. 

print("Training Data Dimensions" + np.shape(x_train))
print("Training Labels Dimensions" + np.shape(y_train))
print("Testing Data Dimensions" + np.shape(x_val))
print("Testing Labels Dimensions" + np.shape(y_val))
print("Bayesian Weight Dimension" + np.shape(weight))

# This is the activation function that will be called during the feedforward step.

def sigmoid(s): 
    return 1/(1 + np.exp(-s))

# This function calculates the derivative for sigmoid and will be called during the backpropagation step.

def sigmoid_derv(s):
    return s * (1 - s)

# This function outputs the probability distribution for diffrent classes. 
 
def softmax(s):
    exps = np.exp(s - np.max(s, axis=1, keepdims=True))
    return exps/np.sum(exps, axis=1, keepdims=True)

# This function outputs the derivative for softmax / error and will be called during the backpropagation step.

def cross_entropy(pred, real):
    n_samples = real.shape[0]
    res = pred - real
    return res/n_samples

# This function outputs the error per epoch. 

def error(pred, real):
    n_samples = real.shape[0]
    logp = - np.log(pred[np.arange(n_samples), real.argmax(axis=1)])
    loss = np.sum(logp)/n_samples
    return loss

class MyNN:
    def __init__(self, x, y):

        self.x = x

        # Modify the number of neurons in the Hidden layer, currently set at 128.

        neurons = 128

        # Modify the Learning Rate, currently set at 0.3. 

        self.lr = 0.3

        ip_dim = x.shape[1]
        op_dim = y.shape[1]

        self.w1 = weight 

        # Random value initialization on the hidden layer weights matrix and bias units. 

        self.b1 = np.zeros((1, 6))
        self.w2 = np.random.randn(6, neurons)
        self.b2 = np.zeros((1, neurons))
        self.w3 = np.random.randn(neurons, op_dim)
        self.b3 = np.zeros((1, op_dim))
        self.y = y

    # The feedforward step does matrix dot product calculation and generates a prediction.

    def feedforward(self):
        z1 = np.dot(self.x, self.w1) + self.b1
        self.a1 = sigmoid(z1)
        z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = sigmoid(z2)
        z3 = np.dot(self.a2, self.w3) + self.b3
        self.a3 = softmax(z3)
    
    # The backprop step calculates the error from the feedforward prediction and makes small adjustments to the weights.

    def backprop(self):
        loss = error(self.a3, self.y)
        print('Error :', loss)
        a3_delta = cross_entropy(self.a3, self.y) # w3
        z2_delta = np.dot(a3_delta, self.w3.T)
        a2_delta = z2_delta * sigmoid_derv(self.a2) # w2
        z1_delta = np.dot(a2_delta, self.w2.T)
        a1_delta = z1_delta * sigmoid_derv(self.a1) # w1

        self.w3 -= self.lr * np.dot(self.a2.T, a3_delta)
        self.b3 -= self.lr * np.sum(a3_delta, axis=0, keepdims=True)
        self.w2 -= self.lr * np.dot(self.a1.T, a2_delta)
        self.b2 -= self.lr * np.sum(a2_delta, axis=0)
        self.w1 -= self.lr * np.dot(self.x.T, a1_delta)
        self.b1 -= self.lr * np.sum(a1_delta, axis=0)

    # This function is solely used for measuring testing accuracy. 

    def predict(self, data):
        self.x = np.array(data)
        self.feedforward()
        return self.a3.argmax()
			

model = MyNN(x_train, np.array(y_train))

# Modify the number of iterations currently set to 500, here:  

epochs = 500

# This step runs the feed forward and backprop set epoch number of times.

for x in range(epochs):
    model.feedforward()
    model.backprop()

# This function is solely used for measuring testing accuracy. 

def get_acc(x, y):
    acc = 0
    for xx,yy in zip(x, y):
        s = model.predict(xx)
        if s == np.argmax(yy):
            acc +=1
    return acc/len(x)*100
	

print("Training accuracy : ", get_acc(x_train, np.array(y_train)))

print("Test accuracy : ", get_acc(x_val, np.array(y_val)))