import streamlit as st
import pandas as pd
import yfinance as yf
from PIL import Image
import requests
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt
from sklearn.preprocessing import  MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import LSTM,Bidirectional,Dense,Dropout
from stocknews import StockNews

st.title('Stock Price Predictions')
st.sidebar.info('Welcome to the Stock Price Prediction App.')

st.markdown("<hr>", unsafe_allow_html=True)
st.subheader("enter the stock ticker")
text = st.text_input('')
st.write(' ')
c1 = yf.Ticker(str(text))
st.header(str(text))

a = c1.info['longBusinessSummary']
st.write(a)

st.markdown("<hr>", unsafe_allow_html=True)
logo = c1.info['website']
url = 'https://logo.clearbit.com/'+str(logo)
response = requests.get(url)
img = Image.open(BytesIO(response.content))
st.header('Logo')
st.image(img)

st.subheader('SECTOR')
sector = c1.info['sector']
st.write(sector)

st.subheader('WEBSITE')
website= c1.info['website']
st.write(website)
st.write("")
st.write("")


StockData= yf.download(str(text),period="max",interval="1d")
st.subheader("Visualiastion of Stock Data")
fig = plt.figure(figsize=(12,6))
plt.plot(StockData['Close'])
st.pyplot(fig)


st.subheader('Moving 200 days mean')
ma_100=StockData.Close.rolling(100).mean()
ma_200=StockData.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(StockData.Close)
plt.plot(ma_100,'r')
plt.plot(ma_200,'g')
st.pyplot(fig)

results,predict,news=st.tabs(["Model Test Results","Prediction","Top Stock News"])

StockData['TradeDate']=StockData.index
FullData=StockData[['Close']].values
sc=MinMaxScaler() 
DataScaler = sc.fit(FullData)
X=DataScaler.transform(FullData)
X_samples = list()
y_samples = list() 
NumerOfRows = len(X)
TimeSteps=10 
for i in range(TimeSteps , NumerOfRows , 1):
    x_sample = X[i-TimeSteps:i]
    y_sample = X[i]
    X_samples.append(x_sample)
    y_samples.append(y_sample)
X_data=np.array(X_samples)
X_data=X_data.reshape(X_data.shape[0],X_data.shape[1], 1)
y_data=np.array(y_samples)
y_data=y_data.reshape(y_data.shape[0], 1)
TestingRecords=5
X_train=X_data[:-TestingRecords]
X_test=X_data[-TestingRecords:]
y_train=y_data[:-TestingRecords]
y_test=y_data[-TestingRecords:]

model = Sequential()
model.add(Bidirectional(LSTM(X_train.shape[1],return_sequences=False), input_shape=(X_train.shape[1],1)))
model.add(Dense(X_train.shape[1]))
model.add(Dense(y_train.shape[1], activation='tanh'))
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
model.fit(X_train, y_train, batch_size = 50, epochs = 100)



predicted_Price = model.predict(X_test)
predicted_Price = DataScaler.inverse_transform(predicted_Price)
orig=y_test
orig=DataScaler.inverse_transform(y_test)

data_d= yf.download(str(text),period="max",interval="1d")
data_d=data_d[['Close']].values

Last10Days=data_d[-10:]
Last10Days=DataScaler.transform(Last10Days.reshape(-1,1))
NumSamples=1
TimeSteps=10
NumFeatures=1
Last10Days=Last10Days.reshape(NumSamples,TimeSteps,NumFeatures)
predicted = model.predict(Last10Days)
predicted = DataScaler.inverse_transform(predicted)



with results:
    st.subheader("Orignal Price Of Stock ")
    st.write (orig)

    st.subheader("Predicted Price Of Stock")
    st.write(predicted_Price)




with news:
    news= StockNews(str(text),save_news=False)
    df_news=news.read_rss()
    n="News of  "+str(text)
    st.header(n)
    for i in range(10):
        st.subheader(f'News {i+1}')
        st.write(df_news['title'][i])
        st.write(df_news['published'][i])
        st.write(df_news['summary'][i])


with predict:
    st.subheader(predicted)
