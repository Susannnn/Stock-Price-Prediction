#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 20:04:11 2020

@author: huangjinghua
"""

import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.metrics import r2_score

#Get the stock quote
df = pd.read_csv('S&P500.csv')


#Select date variable
data = df.filter(['Adj Close'])
#Convert the dataframe to a numpy array
data = data.values
#Get the number of rows to train the model on
training_data_len = math.ceil( len(data) * .8 )


#Scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data)


#Create the training data set
#Create the scaled training data set
data_train = scaled_data[0:training_data_len , :]
#Split the data into x_train and y_train data sets
x_train = []
y_train = []

for i in range(30, len(data_train)):
  x_train.append(data_train[i-30:i, 0])
  y_train.append(data_train[i, 0])

#Convert the x_train and y_train to numpy arrays 
x_train, y_train = np.array(x_train), np.array(y_train)

#Reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_train.shape

#Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape= (x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences= False))
model.add(Dense(25))
model.add(Dense(1))

#Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

#Train the model
model.fit(x_train, y_train, batch_size=10, epochs=50)

#Create the testing data set
data_test = scaled_data[training_data_len - 30: , :]
#Create the data sets x_test and y_test
x_test = []
y_test = scaled_data[training_data_len:, :]
for i in range(30, len(data_test)):
  x_test.append(data_test[i-30:i, 0])

#Convert the data to a numpy array
x_test = np.array(x_test)
 
#Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

#Get the models predicted price values 
predictions = model.predict(x_test, verbose=1)

#Get the root mean squared error (RMSE)
rmse = np.sqrt(np.mean(((predictions - y_test)**2)))
print ('rmse = ', rmse)
std = np.std(predictions - y_test)
print ('std = ', std)
print ('r2 = ', r2_score(y_test, predictions))
