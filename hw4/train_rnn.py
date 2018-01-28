import sys
import os
import re
import json
import numpy as np
from math import floor
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Embedding, LSTM
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

def train(x_all, y_all, valid_set_percentage, weights, batch_size, epochs):
    # Split validation set
    x_train, y_train, x_valid, y_valid = split_valid_set(x_all, y_all, valid_set_percentage)
    
    print(len(x_train), 'train sequences')
    print(len(x_valid), 'valid sequences')
    
    # Build RNN model
    print('Building model...')
    input_size = weights.shape[0]
    output_size = weights.shape[1]
    model = Sequential()
    model.add(Embedding(input_size, output_size, weights=[weights]))
    model.add(LSTM(output_size, dropout=0, recurrent_dropout=0))
    model.add(Dense(output_size, activation='relu'))
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

def word2idx(x_data, dic):
    x_data_idx = []
    for i in range(len(x_data)):
        x_data_idx.append([])
        for j in range(len(x_data[i])):        
            if dic.get(x_data[i][j]) != None:
                x_data_idx[i].append(dic[x_data[i][j]])
            else:
                x_data_idx[i].append(0)
    x_data_idx = np.array(x_data_idx)
    return x_data_idx


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
print('Transfering to index sequence...')
x_train = word2idx(x_train, dic)

# Pad sequences
maxlen = 40
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)

# Train
percentage = 0.1
batch_size = 128
epochs = 20
if not os.path.exists("model"):
    os.mkdir("model")
model = train(x_train, y_train, percentage, weights, batch_size, epochs)
model.save("model/model.h5")