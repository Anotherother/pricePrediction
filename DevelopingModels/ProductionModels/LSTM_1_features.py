import src.get_data as get_data
import src.load_data as load
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import datetime 
from keras import backend as K

#from keras.callbacks import History
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model, save_model

import time


WINDOW = 22


def build_model(input_shape):
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

    
    df = get_data.get_data_frame(typeBlockchain, stock)
    df.index  = df.date
    
    x = df[['close']].copy()
    y = df[['close']].copy()
    
    NUM_FEATURES = x.shape[1]
    
    
    x = pd.ewma(x, 2)
    y = pd.ewma(y, 2)

    # scaling data
    scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    x[['close']] = scaler.fit_transform(x)
    y[['close']] = y_scaler.fit_transform(y)

    x[['cl_2']] = y
    
    # Load data. Split train Test
    #X_train, y_train, X_test, y_test = load.load_data(x, WINDOW, train_size= 0.96, TrainTest = True)
    X_train, y_train = load.load_data(x, WINDOW, TrainTest = False)
    x = x.close ####!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    model = build_model(input_shape=(WINDOW, N))

    # training our model

    print('START FIT MODEL...')
    #history = History()
    #history= model.fit(X_train, y_train, validation_data=(X_test, y_test),  batch_size=32, epochs=500,verbose=0,
    #          callbacks=[history])
    model.fit(X_train, y_train, batch_size=32, epochs=500,verbose=1)
        
    today = time.strftime("_%d_%m_%Y")

    pathModel = "../../models/model_1f_" + typeBlockchain + today +".h5"

    save_model(model, pathModel)

    #model = load_model(pathModel)

    # one day prediction. get last batch known data (now we didnt need in y value and can predict it)    
    lastbatch = np.array(x[-WINDOW:])
    pred = model.predict([lastbatch.reshape(1,WINDOW, NUM_FEATURES)])
    pred =  np.array(y_scaler.inverse_transform(pred)) # predicted value

    # now we make dataframe and create row names in date

    lastDate =str(df.last_valid_index()).split('-')
    currentData = datetime.date(int(lastDate[0]),int(lastDate[1]),int(lastDate[2])) + datetime.timedelta(1)
    predictionDate = pd.date_range(currentData, periods=1)
    prediction = pd.DataFrame(pred, columns=["close"], index = predictionDate)

    print (prediction)

    del model

    K.clear_session()

    return prediction