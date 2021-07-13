import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from matplotlib.pylab import rcParams
rcParams['figure.figsize']=20,10

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))

df=pd.read_csv("NSE-TATA.csv")
df.head()

df["Date"]=pd.to_datetime(df.Date,format="%Y-%m-%d")
df.index=df['Date']

plt.figure(figsize=(16,8))
plt.plot(df["Close"],label='Close Price history')

from keras.models import Sequential
from keras.layers import LSTM,Dense,SimpleRNN

data=df.sort_index(ascending=True,axis=0)
new_dataset=pd.DataFrame(index=range(0,len(df)),columns=['Date','Open','Close'])

for i in range(0,len(data)):
    new_dataset["Date"][i]=data['Date'][i]
    new_dataset["Open"][i]=data["Open"][i]
    new_dataset["Close"][i]=data["Close"][i]
    

new_dataset.index=new_dataset.Date
new_dataset.drop("Date",axis=1,inplace=True)

final_dataset=new_dataset.values

train_data=final_dataset[0:987,:]
valid_data=final_dataset[987:,:]

scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(final_dataset)
x_train_data,y_train_data=[],[]

for i in range(60,len(train_data)):
    x_train_data.append(scaled_data[i-60:i])
    y_train_data.append(scaled_data[i])   

x_train_data,y_train_data=np.array(x_train_data),np.array(y_train_data)
x_train_data=np.reshape(x_train_data,(x_train_data.shape[0],x_train_data.shape[1],2))

rnn_model=Sequential()
rnn_model.add(LSTM(32, input_shape=(x_train_data.shape[1], 2), return_sequences = True))
rnn_model.add(SimpleRNN(32,return_sequences=True))
rnn_model.add(SimpleRNN(32,return_sequences=True))
rnn_model.add(SimpleRNN(32,return_sequences=True))
rnn_model.add(SimpleRNN(32))
rnn_model.add(Dense(2))
rnn_model.compile(loss='mean_squared_error',optimizer='adam')
rnn_model.fit(x_train_data,y_train_data,epochs=10,batch_size=1,verbose=0)

inputs_data=new_dataset[len(new_dataset)-len(valid_data)-60:].values
inputs_data=scaler.transform(inputs_data)
print(inputs_data.shape)

X_test=[]
for i in range(60,inputs_data.shape[0]):
    X_test.append(inputs_data[i-60:i])
X_test=np.array(X_test)
X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],2))

predicted_closing_price=rnn_model.predict(X_test)
predicted_closing_price=scaler.inverse_transform(predicted_closing_price)
print(predicted_closing_price)

rnn_model.save("rnn_open_model.h5")
