import os
import sys
import numpy as np
import pandas as pd

def load_test_data(x_test_path):
    x_test = pd.read_csv(x_test_path, sep=',', header=0)
    x_test = x_test.values
    return x_test

def scaled(x, mean, std):    
    x = (x-mean)/std
    return x

def add_high_order(x, order):
    if order > 1:
        x_tmp = x
        for i in range(2,order+1):
            x_tmp = np.concatenate((x_tmp,x**i), axis=1)
        x = x_tmp
    return x

def sigmoid(z):
    sig = 1 / (1.0 + np.exp(-z))
    
    return np.clip(sig, 1e-8, 1-(1e-8))

def predict(x_test, w, output_path):
    # Add bias
    x_test = np.concatenate((np.ones((len(x_test),1)),x_test),axis=1)
    
    # Predict
    z = np.dot(x_test, w)
    y = sigmoid(z)
    y = np.around(y)    
    
    # Save file
    with open(output_path, 'w') as f:
        f.write('id,label\n')
        for i in range(len(y)):
            f.write('%d,%d\n' %(i+1, y[i]))
    return y

# ----- Main program -----

# Load testing data
x_test_path = sys.argv[1]
x_test = load_test_data(x_test_path)

# Feature scaling
mean = np.load('mean.npy')
std = np.load('std.npy')
x_test = scaled(x_test, mean, std)

# Add high order term
# x_test = add_high_order(x_test, 1)

# Load model
w = np.load('model_generative.npy')

# Predict
predict_path = sys.argv[2]
y_test = predict(x_test, w, predict_path)