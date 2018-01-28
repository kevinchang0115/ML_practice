import sys
import numpy as np
import pandas as pd
from keras.models import load_model

def load_test_data(test_path, col_num, row_num):
    test = pd.read_csv(test_path, sep=',', header=0)
    test = test.values
    data_size = len(test)
    x_test = np.zeros((data_size, col_num, row_num, 1))
    for i in range(data_size):
        data = np.array(test[i,1].split())
        data = data.reshape((col_num, row_num, 1))   
        x_test[i] = data
    x_test /= 255
    return x_test

def numeric(y):
    data_size, class_num = y.shape
    result = np.zeros(data_size)
    for i in range(data_size):
        M = max(y[i])
        for j in range(class_num):
            if M == y[i,j]:
                result[i] = j
    return result

def predict(x_test, model, output_path):   
    # Predict
    y = model.predict(x_test)
    y = numeric(y)
    # Save file
    with open(output_path, 'w') as f:
        f.write('id,label\n')
        for i in range(len(y)):
            f.write('%d,%d\n' %(i, y[i]))
    return y

# ----- Main program -----

# Parameters
row_num = 48
col_num = 48

# Load testing data
test_path = sys.argv[1]
x_test = load_test_data(test_path, col_num, row_num)
#x_test = np.load('x_test.npy')

# Load model
model = load_model('model/model.hdf5')

# Predict
output_path = sys.argv[2]
y_test = predict(x_test, model, output_path)