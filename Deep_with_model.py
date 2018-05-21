import numpy as np
from matplotlib import pyplot
from numpy import concatenate
from math import sqrt

from keras.models import Sequential
from keras.models import load_model

from keras.layers import Dense, Activation
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dropout, Flatten
from keras.layers import LSTM

from keras.utils import np_utils

from copy import deepcopy

#from sklearn.preprocessing import MinMaxScaler
#from sklearn.metrics import mean_squared_error

import pandas as pd
from pandas import DataFrame
from pandas import concat

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



model = load_model('model1.h5')

dataset = pd.read_csv('final_project2018_data/CSV_Create_Plot/beijing_17_18_aq_deep.csv', header=0, index_col=0)
values = dataset.values
values = values.astype('float32')

X_std = values
save_values = deepcopy(values)
for i in range(X_std.shape[1]):
	X_std[:,i] = (values[:,i] - values[:,i].min(axis=0)) / (values[:,i].max(axis=0) - values[:,i].min(axis=0))
scaled = X_std

n_train_hours = 35
n_hours = 48
n_features = 6
n_obs = n_hours * n_features

reframed = series_to_supervised(scaled, n_hours, 1)
values = reframed.values

test = values[n_train_hours:(len(values)-48), :]
test2 = values[(48+n_train_hours):, :]

test_X, test_y = test[:, :n_obs], test2[:, :n_obs]


test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
yplot = model.predict(test_X)

inv_y = yplot
for i in range(X_std.shape[1]):
	inv_y[:,i]  = (inv_y[:,i] * (save_values[:,i].max(axis=0) - save_values[:,i].min(axis=0))) + save_values[:,i].min(axis=0)
yplot = inv_y

for i in range(X_std.shape[1]):
	test_y[:,i]  = (test_y[:,i] * (save_values[:,i].max(axis=0) - save_values[:,i].min(axis=0))) + save_values[:,i].min(axis=0)

for i in [0,1,2,3,4,5]:
	pyplot.plot(yplot[:,i], label='result neural network')
	pyplot.plot(test_y[:,i], label='csv')
	pyplot.legend()
	pyplot.show()
