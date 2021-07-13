import dash
import dash_core_components as dcc
import dash_html_components as html
from matplotlib.pyplot import figure
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np

app = dash.Dash()
server = app.server

scaler=MinMaxScaler(feature_range=(0,1))

df_nse = pd.read_csv("./NSE-TATA.csv")

df_nse["Date"]=pd.to_datetime(df_nse.Date,format="%Y-%m-%d")
df_nse.index=df_nse['Date']
data=df_nse.sort_index(ascending=True,axis=0)

new_data_close=pd.DataFrame(index=range(0,len(df_nse)),columns=['Date','Close'])
new_data_open=pd.DataFrame(index=range(0,len(df_nse)),columns=['Date','Open','Close'])
new_data_high=pd.DataFrame(index=range(0,len(df_nse)),columns=['Date','High','Close'])
new_data_all=pd.DataFrame(index=range(0,len(df_nse)),columns=['Date','Open','High','Close'])

for i in range(0,len(data)):
    new_data_close["Date"][i]=data['Date'][i]
    new_data_close["Close"][i]=data["Close"][i]

    new_data_open["Date"][i]=data['Date'][i]
    new_data_open["Open"][i]=data['Open'][i]
    new_data_open["Close"][i]=data["Close"][i]

    new_data_high["Date"][i]=data['Date'][i]
    new_data_high["High"][i]=data['High'][i]
    new_data_high["Close"][i]=data["Close"][i]

    new_data_all["Date"][i]=data['Date'][i]
    new_data_all["Open"][i]=data['Open'][i]
    new_data_all["High"][i]=data['High'][i]
    new_data_all["Close"][i]=data["Close"][i]

new_data_close.index= new_data_close.Date
new_data_open.index = new_data_open.Date
new_data_high.index = new_data_high.Date
new_data_all.index = new_data_all.Date

new_data_close.drop("Date",axis=1,inplace=True)
new_data_open.drop("Date",axis=1,inplace=True)
new_data_high.drop("Date",axis=1,inplace=True)
new_data_all.drop("Date",axis=1,inplace=True)

dataset_close=new_data_close.values
dataset_open=new_data_open.values
dataset_high=new_data_high.values
dataset_all=new_data_all.values

train_close=dataset_close[0:987,:]
valid_close=dataset_close[987:,:]

train_open=dataset_open[0:987,:]
valid_open=dataset_open[987:,:]

train_high=dataset_high[0:987,:]
valid_high=dataset_high[987:,:]

train_all=dataset_all[0:987,:]
valid_all=dataset_all[987:,:]

scaler_close=MinMaxScaler(feature_range=(0,1))
scaler_open=MinMaxScaler(feature_range=(0,1))
scaler_high=MinMaxScaler(feature_range=(0,1))
scaler_all=MinMaxScaler(feature_range=(0,1))

scaled_data_close= scaler_close.fit_transform(dataset_close)
scaled_data_open = scaler_open.fit_transform(dataset_open)
scaled_data_high = scaler_high.fit_transform(dataset_high)
scaled_data_all = scaler_all.fit_transform(dataset_all)

modelCloseLstm=load_model("lstm_close_model.h5")
modelOpenLstm=load_model("lstm_open_model.h5")
modelHighLstm=load_model('lstm_high_model.h5')
modelAllLstm=load_model("lstm_all_model.h5")

modelCloseRnn=load_model("rnn_close_model.h5")
modelOpenRnn=load_model("rnn_open_model.h5")
modelHighRnn=load_model('rnn_high_model.h5')
modelAllRnn = load_model("rnn_all_model.h5")

inputs_close=new_data_close[len(new_data_close)-len(valid_close)-60:].values
inputs_open=new_data_open[len(new_data_open)-len(valid_open)-60:].values
inputs_high=new_data_high[len(new_data_high)-len(valid_high)-60:].values
inputs_all=new_data_all[len(new_data_all)-len(valid_all)-60:].values

inputs_close=scaler_close.transform(inputs_close)
inputs_open=scaler_open.transform(inputs_open)
inputs_high=scaler_high.transform(inputs_high)
inputs_all=scaler_all.transform(inputs_all)

X_test_close=[]
X_test_open=[]
X_test_high=[]
X_test_all=[]

for i in range(60,inputs_close.shape[0]):
    X_test_close.append(inputs_close[i-60:i])
    X_test_open.append(inputs_open[i-60:i])
    X_test_high.append(inputs_high[i-60:i])
    X_test_all.append(inputs_all[i-60:i])

X_test_close=np.array(X_test_close)
X_test_open=np.array(X_test_open)
X_test_high=np.array(X_test_high)
X_test_all=np.array(X_test_all)

X_test_close=np.reshape(X_test_close,(X_test_close.shape[0],X_test_close.shape[1],1))
X_test_open=np.reshape(X_test_open,(X_test_open.shape[0],X_test_open.shape[1],2))
X_test_high=np.reshape(X_test_high,(X_test_high.shape[0],X_test_high.shape[1],2))
X_test_all=np.reshape(X_test_all,(X_test_all.shape[0],X_test_all.shape[1],3))

closing_price_lstm=modelCloseLstm.predict(X_test_close)
closing_price_lstm=scaler_close.inverse_transform(closing_price_lstm)
open_price_lstm=modelOpenLstm.predict(X_test_open)
open_price_lstm=scaler_open.inverse_transform(open_price_lstm)

closing_price_rnn=modelCloseRnn.predict(X_test_close)
closing_price_rnn=scaler_close.inverse_transform(closing_price_rnn)
open_price_rnn=modelOpenRnn.predict(X_test_open)
open_price_rnn=scaler_open.inverse_transform(open_price_rnn)

high_price_lstm = modelHighLstm.predict(X_test_high)
high_price_lstm = scaler_high.inverse_transform(high_price_lstm)
high_price_rnn = modelHighRnn.predict(X_test_high)
high_price_rnn = scaler_high.inverse_transform(high_price_rnn)

all_price_lstm = modelAllLstm.predict(X_test_all)
all_price_lstm = scaler_all.inverse_transform(all_price_lstm)
all_price_rnn = modelAllRnn.predict(X_test_all)
all_price_rnn = scaler_all.inverse_transform(all_price_rnn)

train=new_data_close[:987]
valid=new_data_close[987:]

app.layout = html.Div([
    html.H1("Stock Price Analysis Dashboard", style={"textAlign": "center"}),
    dcc.Tabs(id="tabs", children=[       
        dcc.Tab(label='LSTM',children=[
			html.Div([
				html.H2("Actual closing price",style={"textAlign": "center"}),
                dcc.Dropdown(id='lstm-dropdown',
                             options=[{'label': 'Close', 'value': 'C'},
                                      {'label': 'Open','value': 'O'},
                                      {'label': 'High','value': 'H'}], 
                             multi=True,value=['C'],
                             style={"display": "block", "margin-left": "auto", 
                                    "margin-right": "auto", "width": "60%"}),
				dcc.Graph(
					id="Actual Data 1",
					figure={
						"data":[
							go.Scatter(
								x=train.index,
								y=valid["Close"],
								mode='lines+markers'
							)
						],
						"layout":go.Layout(
							title='scatter plot',
							xaxis={'title':'Date'},
							yaxis={'title':'Closing Rate'}
						)
					}
				),
				html.H2("LSTM Predicted closing price",style={"textAlign": "center"}),
				dcc.Graph(id="LSTM Predicted Data")				
			])        		
        ]),

        dcc.Tab(label='RNN', children=[
            html.Div([
				html.H2("Actual closing price",style={"textAlign": "center"}),
                dcc.Dropdown(id='rnn-dropdown',
                             options=[{'label': 'Close', 'value': 'C'},
                                      {'label': 'Open','value': 'O'},
                                      {'label': 'High','value': 'H'}], 
                             multi=True,value=['C'],
                             style={"display": "block", "margin-left": "auto", 
                                    "margin-right": "auto", "width": "60%"}),
				dcc.Graph(
					id="Actual Data 2",
					figure={
						"data":[
							go.Scatter(
								x=train.index,
								y=valid["Close"],
								mode='lines+markers'
							)
						],
						"layout":go.Layout(
							title='scatter plot',
							xaxis={'title':'Date'},
							yaxis={'title':'Closing Rate'}
						)
					}
				),
				html.H2("LSTM Predicted closing price",style={"textAlign": "center"}),
				dcc.Graph(id="RNN Predicted Data")				
			])        		
        ])
    ])
])

@app.callback(Output('LSTM Predicted Data', 'figure'),
              [Input('lstm-dropdown', 'value')])
def update_graph(selected_dropdown):
    global valid,closing_price_lstm,open_price_lstm,high_price_lstm,all_price_lstm
    
    print(len(selected_dropdown))
    valid['Predictions'] = closing_price_lstm

    if len(selected_dropdown) == 1:
        if 'O' in selected_dropdown:
            closing_price_predict = []
            for o in open_price_lstm:
                closing_price_predict.append(o[-1])
            valid['Predictions'] = closing_price_predict
        if 'H' in selected_dropdown:
            closing_price_predict = []
            for h in high_price_lstm:
                closing_price_predict.append(h[-1])
            valid['Predictions'] = closing_price_predict

    if len(selected_dropdown) == 2:
        if 'O' in selected_dropdown and 'C' in selected_dropdown:
            closing_price_predict = []
            for o in open_price_lstm:
                closing_price_predict.append(o[-1])
            valid['Predictions'] = closing_price_predict
        if 'H' in selected_dropdown and 'C' in selected_dropdown:
            closing_price_predict = []
            for h in high_price_lstm:
                closing_price_predict.append(h[-1])
            valid['Predictions'] = closing_price_predict
        if 'C' not in selected_dropdown:
            closing_price_predict = []
            for a in all_price_lstm:
                closing_price_predict.append(a[-1])
            valid['Predictions'] = closing_price_predict

    if len(selected_dropdown) == 3:
        closing_price_predict = []
        for a in all_price_lstm:
            closing_price_predict.append(a[-1])
        valid['Predictions'] = closing_price_predict

    figure = go.Figure()
    figure.add_trace(go.Scatter(x=valid.index,y=valid["Predictions"],mode='lines+markers',name='Predictions'))
    figure.add_trace(go.Scatter(x=valid.index,y=valid["Close"],mode='lines+markers',name='Real'))
    figure.update_layout(title='scatter plot',xaxis={"title":'Date'},yaxis={"title":'Closing Rate'})
    return figure

@app.callback(Output('RNN Predicted Data', 'figure'),
              [Input('rnn-dropdown', 'value')])
def update_graph(selected_dropdown):
    global valid,closing_price_rnn,open_price_rnn,high_price_rnn,all_price_rnn
    
    print(len(selected_dropdown))
    valid['Predictions'] = closing_price_rnn

    if len(selected_dropdown) == 1:
        if 'O' in selected_dropdown:
            closing_price_predict = []
            for o in open_price_rnn:
                closing_price_predict.append(o[-1])
            valid['Predictions'] = closing_price_predict
        if 'H' in selected_dropdown:
            closing_price_predict = []
            for h in high_price_rnn:
                closing_price_predict.append(h[-1])
            valid['Predictions'] = closing_price_predict

    if len(selected_dropdown) == 2:
        if 'O' in selected_dropdown and 'C' in selected_dropdown:
            closing_price_predict = []
            for o in open_price_rnn:
                closing_price_predict.append(o[-1])
            valid['Predictions'] = closing_price_predict
        if 'H' in selected_dropdown and 'C' in selected_dropdown:
            closing_price_predict = []
            for h in high_price_rnn:
                closing_price_predict.append(h[-1])
            valid['Predictions'] = closing_price_predict
        if 'C' not in selected_dropdown:
            closing_price_predict = []
            for a in all_price_rnn:
                closing_price_predict.append(a[-1])
            valid['Predictions'] = closing_price_predict

    if len(selected_dropdown) == 3:
        closing_price_predict = []
        for a in all_price_rnn:
            closing_price_predict.append(a[-1])
        valid['Predictions'] = closing_price_predict

    figure = go.Figure()
    figure.add_trace(go.Scatter(x=valid.index,y=valid["Predictions"],mode='lines+markers',name='Predictions'))
    figure.add_trace(go.Scatter(x=valid.index,y=valid["Close"],mode='lines+markers',name='Real'))
    figure.update_layout(title='scatter plot',xaxis={"title":'Date'},yaxis={"title":'Closing Rate'})
    return figure

if __name__=='__main__':
	app.run_server(debug=True)
