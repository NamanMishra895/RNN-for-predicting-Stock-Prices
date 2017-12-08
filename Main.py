#Step 1 - Data Preprocessing
#Import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Import the training set, do not import the test set.

dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

#Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)

#Creating a dataset with 120 timesteps and 1 output
X_train = []
y_train = []
for i in range(120, 1258):
    X_train.append(training_set_scaled[i-120:i,0])
    y_train.append(training_set_scaled[i,0])
X_train, y_train = np.array(X_train), np.array(y_train)
#Reshaping Data
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
 
#Step 2 - Building the RNN
#Importing Libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#Initializing the RNN

regressor = Sequential()
#Adding the first LSTM Layer and Dropout Regularization to avoid Overfitting
regressor.add(LSTM(units=75, return_sequences=True, input_shape= (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))
#Adding the second LSTM Layer and Dropout Regularization to avoid Overfitting
regressor.add(LSTM(units=80, return_sequences=True))
regressor.add(Dropout(0.2))
#Adding the third LSTM Layer and Dropout Regularization to avoid Overfitting
regressor.add(LSTM(units=85, return_sequences=True))
regressor.add(Dropout(0.2))
#Adding the fourth LSTM Layer and Dropout Regularization to avoid Overfitting
regressor.add(LSTM(units=90 , return_sequences=True))
regressor.add(Dropout(0.2))
#Adding the fifth LSTM Layer and Dropout Regularization to avoid Overfitting
regressor.add(LSTM(units=95, return_sequences=True))
regressor.add(Dropout(0.2))
#Adding the sixth LSTM Layer and Dropout Regularization to avoid Overfitting
regressor.add(LSTM(units=100, return_sequences=True))
regressor.add(Dropout(0.2))
#Adding the output layer
regressor.add(Dense(units=1))
#Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
#Fitting the RNN into the training set
regressor.fit(X_train, y_train, epochs=150, batch_size=32)

#Step3 - Making the predictions and visualizing the results
#Getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values
#Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total)-len(dataset_test)- 60 : ].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(120, 80):
    X_test.append(inputs[i-120:i,0])
X_test= np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#Visualize the results
plt.plot(real_stock_price,color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price,color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
