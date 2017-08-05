import src.get_data as get_data
import src.load_data as load
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import datetime 
from keras import backend as K 

from keras.callbacks import History 
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model, save_model

import time
from sklearn import metrics


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
    df = get_data.get_data_frame5minutes(typeBlockchain, stock)
    df = df[['open', 'low', 'high', 'close','volume', 'date_time']][-int(df.shape[0]/ 8 * 3):]
    df.index = df.date_time
    df = df.sort_index()

    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    all_df = df.copy()

    x = all_df[['low', 'high']].copy()

    y = all_df['close'].copy()


    x[['low', 'high']] = x_scaler.fit_transform(x)

    y = y_scaler.fit_transform(y.values.reshape(-1, 1))
    #x['close'] = y
    shape = x.shape[1]
    X_train, y_train = load.load_data(x, WINDOW, TrainTest = False)
    #X_train, y_train, X_test, y_test = load.load_data(x, WINDOW, train_size= 0.90, TrainTest = True)

    model = build_model(input_shape=(WINDOW, shape))

    print('START FIT MODEL...')

    start = time.time()

   # history = History()
    #history= model.fit(X_train, y_train, validation_data=(X_test, y_test),  batch_size=32, epochs=500,verbose=1,
              callbacks=[history])

    model.fit(X_train, y_train, batch_size=128, epochs=200, verbose=1)
    end = time.time()

    print ('Learning time: ', end-start)

    today = time.strftime("_%d_%m_%Y")

    pathModel = "../../models/model_LOWHIGHSHORT_" + typeBlockchain + today +".h5"
    save_model(model, pathModel)
    #model = load_model(pathModel)
    
    # one day prediction. get last batch known data (now we didnt need in y value and can predict it)    
    lastbatch = np.array(x[-WINDOW:])
    pred = model.predict([lastbatch.reshape(1,WINDOW, shape)])
    pred =  np.array(y_scaler.inverse_transform(pred)) # predicted value

    # now we make dataframe and create row names in date

    splitStr = str(df.date_time[df.last_valid_index()]).split(' ')
    lastDate =splitStr[0].split('-')
    lastTime = splitStr[1].split(':')

    predictionDate = datetime.datetime(int(lastDate[0]),int(lastDate[1]),int(lastDate[2]),\
                                    int(lastTime[0]), int(lastTime[1]))  + datetime.timedelta(minutes=5)

    predictionDate = pd.date_range(predictionDate, periods=1)

    prediction = pd.DataFrame(pred, columns=["predictionPrice"], index = predictionDate.values)

    print (prediction)
    
    
    
    del model
    
    K.clear_session()



    
    return prediction