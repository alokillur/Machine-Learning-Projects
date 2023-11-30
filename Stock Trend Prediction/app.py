import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
import wrapt
import h5py
import tensorflow as tf


import fix_yahoo_finance as yf
st.title('Stock Trend Prediction')

user_input= st.text_input('Enter Stock Ticker', 'TSLA')
sData = yf.download(user_input , start = '2013-01-01', end='2023-10-27')

st.subheader('Data from 2013-2023')
st.write(sData.describe())

st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize=(12,6))
plt.plot(sData.Close)
st.pyplot(fig)


st.subheader('Closing Price vs Time chart with 100MA')
ma100 = sData.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(sData.Close)
st.pyplot(fig)


st.subheader('Closing Price vs Time chart with 100MA & 200MA')
ma100 = sData.Close.rolling(100).mean()
ma200 = sData.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100 , 'r')
plt.plot(ma200, 'g')
plt.legend()
plt.plot(sData.Close)
st.pyplot(fig)


data_train = pd.DataFrame(sData['Close'][0:int(len(sData)*.70)])
data_test = pd.DataFrame(sData['Close'][int(len(sData)*.70) : int(len(sData))])


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_train)

model = tf.keras.models.load_model('keras_model.keras')

past_100_days = data_train.tail(100)
final_df = pd.concat([past_100_days, data_test], ignore_index=True)
input_data = scaler.fit_transform(final_df)


x_test = []
y_test =[]

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i,0])

x_test, y_test = np.array(x_test), np.array(y_test)

y_predicted = model.predict(x_test)
scaler = scaler.scale_
scale_factor = 1/scaler[0]

y_predicted = y_predicted* scale_factor
y_test = y_test* scale_factor


st.subheader('Prediction vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label= 'Original Price')
plt.plot(y_predicted, 'r', label= 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)