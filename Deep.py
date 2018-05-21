## all credits to https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/

import numpy as np
#from matplotlib import pyplot
from numpy import concatenate
from math import sqrt

from keras.models import Sequential
from keras.models import load_model

from keras.layers import Dense, Activation
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dropout, Flatten
from keras.layers import LSTM

from copy import deepcopy
from keras.utils import np_utils

#from sklearn.preprocessing import MinMaxScaler
#from sklearn.metrics import mean_squared_error

import pandas as pd
from pandas import DataFrame
from pandas import concat

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


#find on the web
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

# load dataset
dataset = pd.read_csv('final_project2018_data/CSV_Create_Plot/beijing_17_18_aq_deep.csv', header=0, index_col=0)
values = dataset.values
# ensure all data is float
values = values.astype('float32')
# normalize features
X_std = values
save_values = deepcopy(values)
for i in range(X_std.shape[1]):
	X_std[:,i] = (values[:,i] - values[:,i].min(axis=0)) / (values[:,i].max(axis=0) - values[:,i].min(axis=0))


scaled = X_std
print(scaled[885,:])
#scaled = values

n_hours = 48
n_features = 6
# frame as supervised learning
reframed = series_to_supervised(scaled, n_hours, 1)

## split into train and test sets
print(len(values))
values = reframed.values
n_train_hours = 5736
train = values[:(n_train_hours-48), :]
trainPred = values[48:n_train_hours, :]
test = values[n_train_hours:(len(values)-48), :]
testPred = values[(n_train_hours+48):, :]

# split into input and outputs
n_obs = n_hours * n_features
train_X, train_y = train[:, :n_obs], trainPred[:, :n_obs]
test_X, test_y = test[:, :n_obs], testPred[:, :n_obs]

print(train_X.shape, len(train_X), train_y.shape)

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
#train_y = train_y.reshape((train_y.shape[0], n_hours, n_features))
#test_y = test_y.reshape((test_y.shape[0], n_hours, n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
print(train_X.shape[1], train_X.shape[2])

# design network
model = Sequential()
model.add(LSTM(400, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dropout(0.5))
model.add(Dense(288))
model.summary()
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=200, batch_size=800, verbose=2, validation_data=(test_X, test_y) , shuffle=False)

#export loss history for plot later
d = {'col1': history.history['loss'], 'col2': history.history['val_loss']}
df = pd.DataFrame(data=d)
df.to_csv("history.csv")

# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], n_hours*n_features))
inv_yhat = yhat
for i in range(X_std.shape[1]):
	print(save_values[:,i].min(axis=0))
	inv_yhat[:,i]  = (inv_yhat[:,i] * (save_values[:,i].max(axis=0) - save_values[:,i].min(axis=0))) + save_values[:,i].min(axis=0)

#inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 288))
#inv_y = concatenate((test_y, test_X[:, -5:]), axis=1)
inv_y  = test_y
for i in range(X_std.shape[1]):
	inv_y[:,i]  = (inv_y[:,i] * (save_values[:,i].max(axis=0) - save_values[:,i].min(axis=0))) + save_values[:,i].min(axis=0)

rmse2 = np.square(np.subtract(inv_y[:,0], inv_yhat[:,0])).mean()
print('Test RMSE: %.3f' % rmse2)

d = {'col1': inv_yhat[:,0], 'col2': inv_y[:,0], 'col3': inv_yhat[:,1], 'col4': inv_y[:,1], 'col5': inv_yhat[:,2], 'col6': inv_y[:,2], 'col7': inv_yhat[:,3], 'col8': inv_y[:,3], }
df = pd.DataFrame(data=d)
df.to_csv("plotAll.csv")

model.save('model1.h5')
