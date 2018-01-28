import sys
import os
import numpy as np
import pandas as pd
from math import floor
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, BatchNormalization
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, History, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

def load_train_data(train_path, col_num, row_num, class_num):
    train = pd.read_csv(train_path, sep=',', header=0)
    train = train.values     
    data_size = len(train)
    x_train = np.zeros((data_size, col_num, row_num, 1))
    for i in range(data_size):
        data = np.array(train[i, 1].split())
        data = data.reshape((col_num, row_num, 1))
        x_train[i] = data
    x_train /= 255.0
    y_train = to_categorical(train[:,0], class_num)
    
    return (x_train, y_train)

def load_test_data(test_path, col_num, row_num):
    test = pd.read_csv(test_path, sep=',', header=0)
    test = test.values
    data_size = len(test)
    x_test = np.zeros((data_size, col_num, row_num, 1))
    for i in range(data_size):
        data = np.array(test[i,1].split())
        data = data.reshape((col_num, row_num, 1))   
        x_test[i] = data
    x_test /= 255.0
    return x_test    

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

def train(x_all, y_all, valid_set_percentage, input_shape, class_num, batch_size, epochs):
    # Split validation set
    x_train, y_train, x_valid, y_valid = split_valid_set(x_all, y_all, valid_set_percentage)
    # Build CNN model
    model = Sequential()    
    # Conv1
    model.add( Conv2D(64, (5, 5), activation='relu', padding='same', input_shape=input_shape) )   
    model.add( BatchNormalization() )
    model.add( MaxPooling2D((2, 2)) )
    model.add( Dropout(0.2) )
    # Conv2  
    model.add( Conv2D(128, (3, 3), activation='relu', padding='same') )   
    model.add( BatchNormalization() )
    model.add( MaxPooling2D((2, 2)) )
    model.add( Dropout(0.3) )
    # Conv3 
    model.add( Conv2D(256, (3, 3), activation='relu', padding='same') )   
    model.add( BatchNormalization() )
    model.add( MaxPooling2D((2, 2)) )
    model.add( Dropout(0.4) )
    # Conv4
    model.add( Conv2D(512, (3, 3), activation='relu', padding='same') )   
    model.add( BatchNormalization() )
    model.add( MaxPooling2D((2, 2)) )
    model.add( Dropout(0.5) )
    # Flatten
    model.add( Flatten() )    
    # Fc1
    model.add( Dense(1024, activation='relu') )
    model.add( BatchNormalization() )
    model.add( Dropout(0.5) )
	# Fc2
    model.add( Dense(1024, activation='relu') )
    model.add( BatchNormalization() )
    model.add( Dropout(0.5) )
    # Fc3
    model.add( Dense(class_num, activation='softmax') )
	 # Compile model
    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam', metrics=['accuracy'])
    # Show model
    model.summary()
    # Data generator
    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False, 
        featurewise_std_normalization=False, 
        samplewise_std_normalization=False, 
        zca_whitening=False,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2, 
        zoom_range=[0.8, 1.2],
        shear_range=0.2,
        horizontal_flip=True, 
        vertical_flip=False)
    
    datagen.fit(x_train)
    
    # Start training
    early_stop = EarlyStopping(monitor='val_loss', patience=20)
    checkpoint = ModelCheckpoint("model/weights.{epoch:02d}-{val_acc:.2f}.hdf5",
                                 monitor='val_acc', save_best_only=True)
    history = History()    
    model.fit_generator(datagen.flow(x_train, y_train, batch_size = batch_size),
                        steps_per_epoch = len(x_train) // batch_size,
                        epochs = epochs, validation_data = (x_valid, y_valid),
                        callbacks=[history, early_stop, checkpoint])
       
    # Evaluate
    train_score = model.evaluate(x_train, y_train, verbose=0)
    print('\nTrain loss: %f, acc: %f' % (train_score[0], train_score[1]))
    valid_score = model.evaluate(x_valid, y_valid, verbose=0)
    print('\nValid loss: %f, acc: %f' % (valid_score[0], valid_score[1]))
   
    history = history.history
    #plt.plot(np.arange(1,len(history['loss'])+1), history['loss'])
    #plt.plot(np.arange(1,len(history['val_loss'])+1), history['val_loss'])
    #plt.show()
    #plt.plot(np.arange(1,len(history['acc'])+1), history['acc'])
    #plt.plot(np.arange(1,len(history['val_acc'])+1), history['val_acc'])
    #plt.show()
    
    return model


# ----- Main Program -----

# Parameters
row_num = 48
col_num = 48
class_num = 7
input_shape = (col_num, row_num, 1)
valid_set_percentage = 0.1
batch_size = 128
epochs = 128

# Load data
train_path = sys.argv[1]
#test_path = 'data/test.csv'
x_train, y_train = load_train_data(train_path, row_num, col_num, class_num)
#x_test = load_test_data(test_path, row_num, col_num)
#np.save('x_train.npy', x_train)
#np.save('y_train.npy', y_train)
#np.save('x_test.npy', x_test)

#x_train = np.load('x_train.npy')
#y_train = np.load('y_train.npy')
#x_test = np.load('x_test.npy')

# Training
outdir = 'model'
if not os.path.exists(outdir):
    os.makedirs(outdir)
model = train(x_train, y_train, valid_set_percentage, input_shape, class_num, batch_size, epochs)
model.save('model/model_final.hdf5')