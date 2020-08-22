# Recurrent Neural Network



# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import pickle
from tensorflow import keras
from streamlit import caching

# Importing the training set
google_dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
nike_dataset_train = pd.read_csv('Nike_Stock_Price_Train.csv')
pfizer_dataset_train = pd.read_csv('Pfizer_Stock_Price_Train.csv')

company = 'Google'

st.sidebar.header('Choose the Company you want to predict')
def user_input():
    temp = st.sidebar.selectbox('Company', ('Google', 'Nike', 'Pfizer'))
    return temp
    
st.text("Here is a sample data table of what the model is training on: ")

company = user_input()
if(company == 'Google'):
    dataset_train = google_dataset_train
    dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
    training_set = google_dataset_train.iloc[:, 1:2].values
    st.dataframe(google_dataset_train.head())
elif(company == 'Pfizer'):
    dataset_test = pd.read_csv('Pfizer_Stock_Price_Test.csv')
    dataset_train = pfizer_dataset_train
    training_set = pfizer_dataset_train.iloc[:, 1:2].values
    st.dataframe(pfizer_dataset_train.head())
else:
    dataset_test = pd.read_csv('Nike_Stock_Price_Test.csv')
    dataset_train = nike_dataset_train
    training_set = nike_dataset_train.iloc[:, 1:2].values
    st.dataframe(nike_dataset_train)
st.text("Please wait 5-10 minutes for the model to finish training to see the prediction results")
# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, 1232):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#Using Streamlit
caching.clear_cache()
#Header
st.write("""
         # Simple Stock Market Analysis
         """)
latest_iteration = st.empty()
bar = st.progress(0)       
 #Custom Callbacks
class CustomCallback(keras.callbacks.Callback):

   def on_epoch_begin(self, epoch, logs=None):
        keys = list(logs.keys())
        print("End epoch {} of training; got log keys: {}".format(epoch, keys))
        latest_iteration.text(f'Training Progress: {epoch}%')
        bar.progress(epoch)


# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32, callbacks=[CustomCallback()])



# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 82):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
def google_graph():
    plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
    plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price') 
    plt.title('Google Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Google Stock Price')
    plt.legend()
    plt.show()

def nike_graph():
    plt.plot(real_stock_price, color = 'red', label = 'Real Nike Stock Price')
    plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Nike Stock Price') 
    plt.title('Nike Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Nike Stock Price')
    plt.legend()
    plt.show()

def pfizer_graph():
    plt.plot(real_stock_price, color = 'red', label = 'Real Pfizer Stock Price')
    plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Pfizer Stock Price') 
    plt.title('Pfizer Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Pfizer Stock Price')
    plt.legend()
    plt.show()

if(company == 'Google'):
    google_graph()
elif(company == 'Nike'):
    nike_graph()
else:
    pfizer_graph()
st.pyplot()
