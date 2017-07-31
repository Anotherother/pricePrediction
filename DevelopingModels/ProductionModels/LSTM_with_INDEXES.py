import src.get_data as get_data
import src.load_data as load
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import datetime 
from pandas_datareader import data
import datetime as dt
import time 

from keras.callbacks import History 
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model, save_model
from keras import backend as K

import time
from sklearn import metrics

WINDOW = 22

def IdexDataframe(name):
    start = dt.datetime(2015,8,1)
    end = dt.datetime.today()
    
    INDEXValue = data.DataReader("^"+name, 'yahoo', start, end)
    INDEXValue = pd.DataFrame(data =INDEXValue, index=pd.DatetimeIndex(start=start,
                    end=dt.datetime.today(), freq='D'))
    
    INDEXValue.columns = ['Open' + name, 'High' + name,\
                                        'Low' + name,  'Close' + name,\
                                        'Adj Close' + name,'Volume' + name]
    INDEXValue = INDEXValue.fillna(method='ffill')                   
    return INDEXValue

def build_model(input_shape):
    
    # LSTM NN
    d = 0.2
    model = Sequential()
    
    model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(d))
        
    model.add(LSTM(128, input_shape=input_shape, return_sequences=False))
    model.add(Dropout(d))
        
    model.add(Dense(32,kernel_initializer="normal",activation='relu'))        
    model.add(Dense(1,kernel_initializer="normal",activation='linear'))
    
    model.compile(loss='mse',optimizer='adam', metrics=['accuracy'])
    
    return model


def nextDayPrediction(typeBlockchain, stock):    
    """
    Triggers for plotting
    """
    
    DJA = IdexDataframe("DJA")
    GSPC = IdexDataframe("GSPC")
    IXK = IdexDataframe("IXK")

    loaded = get_data.get_data_frame()
    loaded.index = loaded.date

    loaded = loaded[['open', 'close', 'low', 'high', 'volume']]
    df = pd.concat([DJA,GSPC, IXK, loaded], axis = 1, ignore_index=False)
    df = df.fillna(method='ffill')
    df = df.dropna(axis = 1, thresh=10)
    df = df.dropna(axis = 0, thresh=10)
    df = df[:-1]
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    all_df = df.copy()

    feature = ['OpenDJA', 'HighDJA', 'LowDJA', 'CloseDJA', 'Adj CloseDJA', 'VolumeDJA',
       'OpenGSPC', 'HighGSPC', 'LowGSPC', 'CloseGSPC', 'Adj CloseGSPC',
       'VolumeGSPC', 'open', 'low', 'high', 'volume']

    x = all_df[feature].copy()
    y = all_df['close'].copy()

    x = pd.ewma(x,2)
    y = pd.ewma(y,2)
    
    x[feature] = x_scaler.fit_transform(x)

    y = y_scaler.fit_transform(y.values.reshape(-1, 1))
    x['close'] = y
    
    num_features = x.shape[1]
    X_train, y_train = load.load_data(x, WINDOW, TrainTest = False)
    
    model = build_model(input_shape=(WINDOW, num_features))
    
    print('START FIT MODEL...')
    
    start = time.time()
    
    model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=1)
    end = time.time()

    print ('Learning time: ', end-start)
    
    today = time.strftime("_%d_%m_%Y")
    
    pathModel = "../../models/model_5f_" + typeBlockchain + today +".h5"
    save_model(model, pathModel)
    #model = load_model(pathModel)
    # one day prediction. get last batch known data (now we didnt need in y value and can predict it)    
    lastbatch = np.array(x[-WINDOW:])
    pred = model.predict([lastbatch.reshape(1,22, num_features)])
    pred =  np.array(y_scaler.inverse_transform(pred)) # predicted value
    prediction = pred.reshape(-1)

    K.clear_session()
    
    print (prediction)

    return prediction