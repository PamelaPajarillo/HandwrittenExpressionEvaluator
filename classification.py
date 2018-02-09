
from __future__ import print_function
import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import cv2
import numpy as np
import random

import os
import os.path
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)

CLASSES = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'plus', 'minus', 'multiplication', 'division1']
NUM_CLASSES = len(CLASSES)  #The number of classes
IMG_SIZE = 28
BATCH_SIZE = 128	        # The number of images to process during a single pass
EPOCHS = 5000	                # The number of times to iterate through the entire training set
IMG_ROWS, IMG_COLS = IMG_SIZE, IMG_SIZE                 # Input Image Dimensions
DATA_UTILIZATION = 1        # Fraction of data which is utilized in training and testing
TEST_RATIO = 1/6

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename)) # Reads the images from folder
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)     # Converts images to black and white
        if img is not None:
            images.append(img)
    return images

# Define function to load data set and return testing and training data
def loadData(test_ratio = TEST_RATIO):
    data = np.zeros((0,IMG_ROWS, IMG_COLS))
    labels = []

    ###Load data from folder titled after the character class name eg. ('Zero', 'One', 'Two'...) and label it with a
    ###corresponding integer value eg. (0, 1, 2...)
    for i, CLASS in enumerate(CLASSES):
        images = load_images_from_folder(CLASS)       # Load images from folder
        print(CLASS +" loaded")

        data = np.concatenate((data,images), axis=0)    # Add features (images) to data variable
        print("concatenated")

        label = len(images) * [i]                           # Create a list of the feature labels the length of the number of images
        labels = np.concatenate((labels, label), axis=0)    # Append the list of labels to the labels variable

    sort_data = list(zip(data,labels))                  # Zip the together the labels and features
    random.shuffle(sort_data)                           # Shuffle the labels and features together
    data, labels = zip(*sort_data)                      # Unzip the labels-features variable back into data and labels

    # Delete proportion of data equal to 1-DATA_UTILIZATION to speed up training and testing
    data = data[0:int(len(data)*DATA_UTILIZATION)]
    labels = labels[0:int(len(data)*DATA_UTILIZATION)]

    # Split data into test and train sets
    cutoff = int((1 - TEST_RATIO) * len(data))          # Determine the index at which to split the dataset into test and train
    x_train = data[0:cutoff]                            # Training features
    y_train = labels[0:cutoff]                          # Training labels
    x_test = data[cutoff:]                              # Testing features
    y_test = labels[cutoff:]                            # Testing labels

    return (np.asarray(x_train), np.asarray(y_train)), (np.asarray(x_test), np.asarray(y_test))

# Load data
(x_train, y_train), (x_test, y_test) = loadData(test_ratio = TEST_RATIO)

x_test = x_test.reshape(x_test.shape[0],IMG_ROWS, IMG_COLS,1)     # Reshape x_test where 1 = number of colors
x_train = x_train.reshape(x_train.shape[0],IMG_ROWS, IMG_COLS,1)  # Reshape x_test
input_shape = (IMG_ROWS, IMG_COLS,1)
x_train = x_train.astype('float32')     # Convert x_train to float32
x_test = x_test.astype('float32')       # Convert x_test to float 32

x_train /= 255      # Scale feature values from 0-255 to values from 0-1
x_test /= 255       # Scale feature values from 0-255 to values from 0-1

print('x_train shape: ', x_train.shape)
print('y_train shape: ', y_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to train = keras.utils.to_categorical(y_train,NUM_CLASSES = None) to binary class matrices
# Arguments: y: Class vector to be converted into a matrix (integers from 0 to num_classes)
#           num_classes: total number of classes
y_train = keras.utils.to_categorical(y_train,num_classes = None)
y_test = keras.utils.to_categorical(y_test, num_classes = None)


#Define network
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
# model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dense(NUM_CLASSES, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)

model.save('classification')
print('Test loss:', score[0])
print('Test accuracy:', score[1])
