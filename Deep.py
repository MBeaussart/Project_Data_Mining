## all credits to https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/

import numpy as np
#from matplotlib import pyplot
from numpy import concatenate
from math import sqrt

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dropout, Flatten
from keras.layers import LSTM

from keras.utils import np_utils

#from sklearn.preprocessing import MinMaxScaler
#from sklearn.metrics import mean_squared_error

import pandas as pd
from pandas import DataFrame
from pandas import concat


import tensorflow as tf
from keras import backend as K

num_cores = 4

num_GPU = 2
num_CPU = 5

config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,\
		inter_op_parallelism_threads=num_cores, allow_soft_placement=True,\
		device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})
session = tf.Session(config=config)
K.set_session(session)

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
#scaler = MinMaxScaler(feature_range=(0, 1))
#scaled = scaler.fit_transform(values)
#X_std = (values - values.min(axis=0)) / (values.max(axis=0) - values.min(axis=0))
#scaled = X_std * (1 - 0) + 0
scaled = values


n_hours = 48
n_features = 6
# frame as supervised learning
reframed = series_to_supervised(scaled, n_hours, 1)


# split into train and test sets
values = reframed.values
n_train_hours = 5736
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
# split into input and outputs
n_obs = n_hours * n_features
train_X, train_y = train[:, :n_obs], train[:, :n_obs]
test_X, test_y = test[:, :n_obs], test[:, :n_obs]
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
model.add(LSTM(288, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(288,input_shape=(train_X.shape[1], train_X.shape[2])))
model.summary()
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=70, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
#pyplot.plot(history.history['loss'], label='train')
#pyplot.plot(history.history['val_loss'], label='test')
#pyplot.legend()
#pyplot.show()





# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], n_hours*n_features))
# invert scaling for forecast
#inv_yhat = concatenate((yhat, test_X[:, -5:]), axis=1)
inv_yhat = yhat
#inv_yhat  = (inv_yhat * (inv_yhat.max(axis=0) - inv_yhat.min(axis=0))) + inv_yhat.min(axis=0)

#inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 288))
#inv_y = concatenate((test_y, test_X[:, -5:]), axis=1)
inv_y  = test_y
#inv_y  = (inv_y * (inv_y.max(axis=0) - inv_y.min(axis=0))) + inv_y.min(axis=0)

#inv_y = inv_y[:,0]
# calculate RMSE
#rmse = sqrt(mean_squared_error(inv_y[:,0], inv_yhat[:,0]))
#print('Test RMSE: %.3f' % rmse)
rmse2 = np.square(np.subtract(inv_y[:,0], inv_yhat[:,0])).mean()
print('Test RMSE: %.3f' % rmse2)

#for i in [0,1,2,3,4,5]:
	#pyplot.plot(inv_yhat[:,i])
	#pyplot.plot(inv_y[:,i])
	#pyplot.show()
