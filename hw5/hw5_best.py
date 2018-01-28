import numpy as np
import pandas as pd
import os
import sys
from keras.models import load_model

def load_test_data(test_path):
    data = pd.read_csv(test_path, header=0)
    x_test_users = data['UserID'].values
    x_test_movies = data['MovieID'].values
    return x_test_users, x_test_movies

def denormalize(y, method='Standarization'):
    if method == 'Standarization':
        mean = np.load('model/y_rating_mean.npy')
        std = np.load('model/y_rating_std.npy')
        return y*std+mean
    elif method == 'Feature scaling':
        return y*4+1
    else:
        print('Error: No normalization method found...')
        return y
       

#----- Main Program -----

# Load test data
print('Loading test data...')
test_path = sys.argv[1]
x_test_users, x_test_movies = load_test_data(test_path)

# Load model
print('Loading model...')
model_name1 = 'mf.h5'
model_name2 = 'nn.h5'
model_path1 = os.path.join('model', model_name1)
model_path2 = os.path.join('model', model_name2)
model1 = load_model(model_path1)
model2 = load_model(model_path2)

# Predict
print('Predicting results...')
method = 'Standarization'
#method = 'Feature scaling'
#method = ''
y1 = model1.predict([x_test_users, x_test_movies])
y2 = model2.predict([x_test_users, x_test_movies])
y1 = denormalize(y1, method)
y2 = denormalize(y2, method)
y = (y1+y2)/2
# Save file
print('Saving results...')
output_path = sys.argv[2]
with open(output_path, 'w') as f:
    f.write('TestDataID,Rating\n')
    for i in range(len(y)):
        f.write('%d,%f\n' %(i+1, y[i]))
