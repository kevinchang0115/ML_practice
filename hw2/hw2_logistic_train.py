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

def add_high_order(x, order):
    if order > 1:
        x_tmp = x
        for i in range(2,order+1):
            x_tmp = np.concatenate((x_tmp,x**i), axis=1)
        x = x_tmp
    return x

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
    
def train(x_all,y_all):
    # Add bias
    x_all = np.concatenate((np.ones((len(x_all),1)),x_all), axis=1)
    
    # Split validation set
    valid_set_percentage = 0.1
    x_train, y_train, x_valid, y_valid = split_valid_set(x_all, y_all, valid_set_percentage)
    train_data_size = len(x_train)
    
    # Initialize weight
    w = np.zeros(len(x_train[0]))
    
    # Initialize adam parameters
    l_rate = 0.001
    m = np.zeros(len(x_train[0]))
    v = np.zeros(len(x_train[0]))
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    
    # Set batch parameters
    batch_size = 1000
    step_num = int(floor(train_data_size / batch_size))
    
    # Set epoch parameters
    epoch_num = 1000
    evaluate_iter = 50

    # Start training
    total_loss = 0.0
    for epoch in range(1,epoch_num+1):
        # Evaluate loss and accuracy
        if (epoch) % evaluate_iter == 0:
            print('----- Epoch num = %d -----' % epoch )
            print('Epoch avg loss = %f' % (total_loss / (float(evaluate_iter) * train_data_size)))
            total_loss = 0.0
            evaluate(x_train, y_train, w, 'Train')
            evaluate(x_valid, y_valid, w, 'Valid')
            print()

        # Random shuffle
        x_train, y_train = shuffle(x_train, y_train)

        # Train with batch
        for i in range(step_num):
            x_batch = x_train[i*batch_size:(i+1)*batch_size]
            y_batch = y_train[i*batch_size:(i+1)*batch_size]
            z = np.dot(x_batch, w)
            y = sigmoid(z)
            
            # Calculate loss function
            cross_entropy = -1 * (np.dot(np.squeeze(y_batch), np.log(y)) + np.dot((1-np.squeeze(y_batch)), np.log(1-y)))
            total_loss += cross_entropy
            
            grad = -np.dot( np.transpose(x_batch), np.squeeze(y_batch)-y )
    
            # Update adam parameters
            m = beta1*m+(1-beta1)*grad
            v = beta2*v+(1-beta2)*grad**2
            m_h = m/(1-beta1**(i+1))
            v_h = v/(1-beta2**(i+1))
            adam = m_h/(np.sqrt(v_h)+eps)
            
            # Update weight
            w = w - l_rate * adam
    
    return w

# ----- Main program -----

# Load data
x_train, y_train = load_train_data('data/X_train', 'data/Y_train')
x_test = load_test_data('data/X_test')

# Feature scaling
x_train, x_test, mean, std = feature_scaling(x_train, x_test)
np.save('mean.npy', mean)
np.save('std.npy', std)

# Add high order term
x_train = add_high_order(x_train, 2)

# Training
w = train(x_train, y_train)

# Save model
np.save('model_logistic.npy', w)