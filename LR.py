import math
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score


df = pd.read_csv('S&P500.csv')

#Select date variable
data = df.filter(['Adj Close'])

data = data.values

#Get the number of rows to train the model on
training_data_len = math.ceil( len(data) * .8 )


scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data)

data_train = scaled_data[0:training_data_len , :]


x_train = []
y_train = []

for i in range(5, len(data_train)):
  x_train.append(data_train[i-5:i, 0])
  y_train.append(data_train[i, 0])
  
x_train, y_train = np.array(x_train), np.array(y_train)

linear_reg = LinearRegression()
# Train the model
linear_reg.fit(x_train, y_train)

#Create the testing data set
data_test = scaled_data[training_data_len - 5: , :]
#Create the data sets x_test and y_test
x_test = []
y_test = scaled_data[training_data_len:, :]
for i in range(5, len(data_test)):
  x_test.append(data_test[i-5:i, 0])

#Convert the data to a numpy array
x_test = np.array(x_test)
y_test = np.array(y_test)

predictions = linear_reg.predict(x_test)
rmse = np.sqrt(np.mean(((predictions - y_test)**2)))
print ('rmse = ', rmse)
std = np.std(predictions - y_test)
print ('std = ', std)
print ('r2 = ', r2_score(y_test, predictions))
