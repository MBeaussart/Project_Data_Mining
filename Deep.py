
#============================

#just an example of code for kaggle competiton, https://www.kaggle.com/krashski/convolutional-neural-network-with-keras/code

#=============================
import numpy as np
np.random.seed(0)  #for reproducibility

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dropout, Flatten

from keras.utils import np_utils

input_size = 784
batch_size = 200
hidden_neurons = 200
classes = 10
epochs = 6

import pandas as pd

# Read data
train = pd.read_csv('train.csv')
labels = train.ix[:,0].values.astype('int32')
X_train = (train.ix[:,1:].values).astype('float32')
X_test = (pd.read_csv('test.csv').values).astype('float32')

print (X_train.shape)


# reshape the input data
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

print(X_train[0].shape)

# convert list of labels to binary class matrix
y_train = np_utils.to_categorical(labels, classes)


input_dim = X_train.shape[1]
nb_classes = classes

# create CNN model
model = Sequential()
model.add(Convolution2D(32, (3, 3), input_shape=(28, 28, 1)))
model.add(Activation('relu'))
model.add(Convolution2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(hidden_neurons))
model.add(Activation('relu'))
model.add(Dense(classes))
model.add(Activation('softmax'))


model.compile(loss='categorical_crossentropy',
	metrics=['accuracy'], optimizer='adadelta')

print('Training...')
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)

print("Generating test predictions...")
preds = model.predict_classes(X_test, verbose=0)

def write_preds(preds, fname):
	pd.DataFrame({"ImageId": list(range(1,len(preds)+1)), "Label": preds}).to_csv(fname, index=False, header=True)

write_preds(preds, "keras-cnn.csv")
