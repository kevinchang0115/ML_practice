import numpy as np
import pandas as pd
from math import floor

def load_train_data(x_train_path,y_train_path):
    x_train = pd.read_csv(x_train_path, sep=',', header=0)
    x_train = x_train.values
    y_train = pd.read_csv(y_train_path, sep=',', header=0)
    y_train = y_train.values
    return (x_train, y_train)

def load_test_data(x_test_path):
    x_test = pd.read_csv(x_test_path, sep=',', header=0)
    x_test = x_test.values
    return x_test

def normalize(x):
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    x = (x-mean)/std
    return (x, mean, std)

def feature_scaling(x_train, x_test):
    x_all = np.concatenate( (x_train, x_test), axis=0 )
    x_all, mean, std = normalize(x_all)
    x_train = x_all[:len(x_train)]
    x_test = x_all[len(x_train):]
    return (x_train, x_test, mean, std)

def shuffle(x, y):
    rand = np.arange(len(x))
    np.random.shuffle(rand)
    return (x[rand], y[rand])

def split_valid_set(x_all, y_all, percentage):
    all_data_size = len(x_all)
    valid_data_size = int(floor(all_data_size * percentage))

    x_all, y_all = shuffle(x_all, y_all)

    x_valid, y_valid = x_all[0:valid_data_size], y_all[0:valid_data_size]
    x_train, y_train = x_all[valid_data_size:], y_all[valid_data_size:]    
    
    return (x_train, y_train, x_valid, y_valid)

def sigmoid(z):
    sig = 1 / (1.0 + np.exp(-z))
    return np.clip(sig, 1e-8, 1-(1e-8))

def evaluate(x_eval, y_eval, w, group):
    data_size = len(x_eval)
    z = np.dot(x_eval,w)
    y = sigmoid(z)
    y = np.around(y)
    result = (np.squeeze(y_eval) == y)
    print( group + ' acc = %f' % (float(result.sum()) / data_size) )
    return    

def gaussian(x_train, y_train):
    # Initialize
    train_data_size, feature_size = x_train.shape
    cnt1 = 0
    cnt2 = 0    
    mu1 = np.zeros(feature_size)
    mu2 = np.zeros(feature_size)
    sigma1 = np.zeros((feature_size,feature_size))
    sigma2 = np.zeros((feature_size,feature_size))
    
    # Calculate parameters
    for i in range(train_data_size):
        if y_train[i] == 1:
            cnt1 += 1
            mu1 += x_train[i]
        else:
            cnt2 += 1
            mu2 += x_train[i]
    mu1 /= cnt1
    mu2 /= cnt2
    
    for i in range(train_data_size):
        if y_train[i] == 1:
            sigma1 += np.dot(np.transpose([x_train[i] - mu1]), [(x_train[i] - mu1)])
        else:
            sigma2 += np.dot(np.transpose([x_train[i] - mu2]), [(x_train[i] - mu2)])
    sigma1 /= cnt1
    sigma2 /= cnt2
    shared_sigma = float(cnt1) / train_data_size * sigma1 + float(cnt2) / train_data_size * sigma2
    
    return (cnt1, cnt2, mu1, mu2, shared_sigma)
    
def train(x_all, y_all):    
    # Split validation set
    valid_set_percentage = 0.1
    x_train, y_train, x_valid, y_valid = split_valid_set(x_all, y_all, valid_set_percentage)
    train_data_size, feature_size = x_train.shape
    
    # Calculate Gaussian model parameters
    cnt1, cnt2, mu1, mu2, shared_sigma = gaussian(x_train, y_train)
    sigma_inv = np.linalg.inv(shared_sigma)
    
    # Calculate weight and bias
    w = np.dot(mu1-mu2, sigma_inv)
    b = (-0.5)*np.dot(np.dot([mu1],sigma_inv),mu1) + (0.5)*np.dot(np.dot([mu2],sigma_inv),mu2) + np.log(float(cnt1)/cnt2)
    
    # Concatenate weight and bias
    w = np.concatenate((b,w))
    x_train = np.concatenate( (np.ones((len(x_train),1)),x_train), axis=1 )
    x_valid = np.concatenate( (np.ones((len(x_valid),1)),x_valid), axis=1 )
   
    # Evaluate accuracy
    evaluate(x_train, y_train, w, 'Train')
    evaluate(x_valid, y_valid, w, 'Valid')

    return w

# ----- Main program -----

# Load data
x_train, y_train = load_train_data('data/X_train', 'data/Y_train')
x_test = load_test_data('data/X_test')

# Feature scaling
x_train, x_test, mean, std = feature_scaling(x_train, x_test)
np.save('mean.npy', mean)
np.save('std.npy', std)

# Training
w = train(x_train, y_train)

# Save model
np.save('model_generative.npy', w)