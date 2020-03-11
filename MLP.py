#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 21:03:15 2020

@author: huangjinghua
"""
import math
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

df = pd.read_csv('S&P500.csv')

#Select date variable
data = df.filter(['Adj Close'])

#Make data a np.array
data = data.values

training_data_len = math.ceil( len(data) * .8 )

#Rescale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data)

#Create training set and test set 
data_train = scaled_data[0:training_data_len , :]
data_test = scaled_data[training_data_len - 5: , :]

#Build x_train and y_train
x_train = []
y_train = []

for i in range(5, len(data_train)):
  x_train.append(data_train[i-5:i, 0])
  y_train.append(data_train[i, 0])
  
x_train, y_train = np.array(x_train), np.array(y_train)



#Build the MLP model
model = Sequential()
model.add(Dense(10, input_dim=5, activation='relu'))  
model.add(Dense(5, activation='relu'))  
model.add(Dense(1, activation='linear'))   

#Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

#Train the model
model.fit(x_train, y_train, batch_size=10, epochs=50)

#Create  x_test and y_test
x_test = []
y_test = scaled_data[training_data_len:, 0]
for i in range(5, len(data_test)):
  x_test.append(data_test[i-5:i, 0])


#Convert the data to a numpy array
x_test = np.array(x_test)

#Get the models predicted price values 
predictions = model.predict(x_test, verbose=1)

#Get the root mean squared error (RMSE)
rmse = np.sqrt(np.mean(((predictions - y_test)**2)))
print ('rmse = ', rmse)

#Get the standard deviation
std = np.std(predictions - y_test)
print ('std = ', std)

#Get R Squared
print ('r2 = ', r2_score(y_test, predictions))

