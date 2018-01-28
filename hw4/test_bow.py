import sys
import re
import json
import numpy as np
from keras.models import load_model

def load_test_data(test_path, with_marks=True):
    file = open(test_path, 'r+', encoding = 'utf8')
    next(file)
    x_test = []
    for row in file:
        text = row.split(',', 1)
        if with_marks:
            x_test.append(re.findall(r'[A-Za-z0-9,.:;?!\'"()\[\]{}<>*/+-=~_#$%&]+',text[1]))
        else:
            x_test.append(re.findall(r'[A-Za-z0-9]+',text[1]))                                    
    x_test = np.array(x_test)
    return x_test

def predict(x_test, model, output_path):
    # Predict
    print('Predicting results...')
    y = model.predict(x_test, verbose=1)
    y = np.round(y)
    # Save file
    print('Saving results...')
    with open(output_path, 'w') as f:
        f.write('id,label\n')
        for i in range(len(y)):
            f.write('%d,%d\n' %(i, y[i]))
    return y

def word2bow(x_data, dic, nb_word):
    x_data_bow = []
    for i in range(len(x_data)):
        x_data_bow.append(np.zeros(nb_word, dtype='uint8'))
        for j in range(len(x_data[i])):            
            if dic.get(x_data[i][j]) != None:
                x_data_bow[i][dic[x_data[i][j]]] = x_data_bow[i][dic[x_data[i][j]]] + 1
            else:
                x_data_bow[i][0] + 1
    return x_data_bow


#----- Main Program -----

# Load test data
print('Loading test data')
test_path = sys.argv[1]
x_test = load_test_data(test_path)

# Load dictionary
print('Loading dictionary...')
weights = np.load("dictionary/weights_128_semi.npy")
with open("dictionary/dict_128_semi", 'r') as f:
    dic = json.loads(f.read())
x_test = word2bow(x_test, dic, len(weights))
x_test = np.array(x_test)

# Load model
print('Loading model...')
model_path = 'model/best_model.h5'
model = load_model(model_path)

# Predict
output_path = sys.argv[2]
y_test = predict(x_test, model, output_path)