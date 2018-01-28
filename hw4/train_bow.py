import sys
import os
import re
import json
import numpy as np
from math import floor
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, History, ModelCheckpoint
import matplotlib.pyplot as plt

def load_train_data(train_path, with_marks=True):
    file = open(train_path, 'r+', encoding = 'utf8')
    x_train = []
    y_train = []
    for row in file:
        text = row.split(' +++$+++ ')
        if with_marks:
            x_train.append(re.findall(r'[A-Za-z0-9,.:;?!\'"()\[\]{}<>*/+-=~_#$%&]+',text[1]))
        else:
            x_train.append(re.findall(r'[A-Za-z0-9]+',text[1]))
        y_train.append(text[0])
    x_train = np.array(x_train)
    y_train = np.array(y_train)        
    return x_train, y_train

def shuffle(x, y):
    #np.random.seed(10)
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

def train(x_all, y_all, valid_set_percentage, batch_size, epochs):
    # Split validation set
    x_train, y_train, x_valid, y_valid = split_valid_set(x_all, y_all, valid_set_percentage)
    
    print(len(x_train), 'train sequences')
    print(len(x_valid), 'valid sequences')
    
    # Build RNN model
    print('Building model...')
    model = Sequential()
    model.add(Dense(128, input_dim=x_train.shape[1], activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Show model
    model.summary()
    
    # Start training
    print('Training...')
    early_stop = EarlyStopping(monitor='val_loss', patience=2)
    checkpoint = ModelCheckpoint("model/weights.{val_loss:.3f}-{val_acc:.3f}.h5",
                                 monitor='val_loss', save_best_only=True)
    history = History()
    model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs,
                        validation_data = (x_valid, y_valid), 
                        callbacks=[early_stop, checkpoint, history])
    # Evaluate
    train_score = model.evaluate(x_train, y_train, verbose=0)
    print('\nTrain loss: %f, acc: %f' % (train_score[0], train_score[1]))
    valid_score = model.evaluate(x_valid, y_valid, verbose=0)
    print('\nValid loss: %f, acc: %f' % (valid_score[0], valid_score[1]))
    
    # Plot
    history = history.history
    plt.plot(np.arange(1,len(history['loss'])+1), history['loss'])
    plt.plot(np.arange(1,len(history['val_loss'])+1), history['val_loss'])
    plt.show()
    plt.plot(np.arange(1,len(history['acc'])+1), history['acc'])
    plt.plot(np.arange(1,len(history['val_acc'])+1), history['val_acc'])
    plt.show()
    
    return model

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

# Load train data
print('Loading train data...')
train_path = sys.argv[1]
x_train, y_train = load_train_data(train_path)

# Load dictionary
print('Loading dictionary...')
weights = np.load("dictionary/weights_128_semi.npy")
with open("dictionary/dict_128_semi", 'r') as f:
    dic = json.loads(f.read())
print('Transfering to BOW...')
x_train = word2bow(x_train, dic, len(weights))
x_train = np.array(x_train)

# Train
percentage = 0.1
batch_size = 128
epochs = 20
if not os.path.exists("model"):
    os.mkdir("model")
model = train(x_train, y_train, percentage, batch_size, epochs)
model.save("model/model.h5")