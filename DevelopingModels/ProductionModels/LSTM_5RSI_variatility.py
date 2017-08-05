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

def variability(feature):
    w = 96
    
    length = int(len(feature)/w) + 1
    s = np.zeros(length)

    for j in np.arange (1,length+ 1 ):
        
        d = feature[j*w - w: j * w]

        for i in np.arange(1, d.shape[0]):
            s[j-1] += np.abs(d[i] - d[i-1])

    return s

def getViriabilityDataframe(typeBlockchain, stock):
    df = get_data.get_data_frame5minutes(typeBlockchain, stock)

    computed_variability = variability(df.volume.values) 
    variabilityDataframe= pd.DataFrame(computed_variability[1:-1].T, index= df.date.unique()[1:-1], columns=['variability'])
    variabilityDataframe.index.names = ['date']
    
    return variabilityDataframe

def computeRSI(data):
    window_length = 9
    close = data['close'] # Get just the close

    # Get the difference in price from previous step
    delta = close.diff()
    # Get rid of the first row, which is NaN since it did not have a previous 
    # row to calculate the differences
    delta = delta[1:] 

    # Make the positive gains (up) and negative gains (down) Series
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0

    # Calculate the EWMA
    roll_up1 = pd.ewma(up, window_length)
    roll_down1 = pd.ewma(down.abs(), window_length)

    # Calculate the RSI based on EWMA
    RS1 = roll_up1 / roll_down1
    RSI1 = 100.0 - (100.0 / (1.0 + RS1))
    RSI1.index = data.date[1:]
    #RSI1 = pd.DataFrame(data = RSI1, index = data.date[1:])
    
    # Calculate the SMA
    roll_up2 = pd.rolling_mean(up, window_length)
    roll_down2 = pd.rolling_mean(down.abs(), window_length)

    # Calculate the RSI based on SMA
    RS2 = roll_up2 / roll_down2
    RSI2 = 100.0 - (100.0 / (1.0 + RS2))
    RSI2.index = data.date[1:]
    #RSI2 = pd.DataFrame(data = RSI2, index = data.date[1:])
    return RSI1, RSI2


def nextDayPrediction(typeBlockchain, stock):    

    
    variab = getViriabilityDataframe(typeBlockchain, stock)

    df = get_data.get_data_frame(typeBlockchain, stock)
    df.index = df.date
    RSI1, RSI2 = computeRSI(df)
    
    featurevector  = pd.concat([df.date, RSI1, RSI2, variab, df.close, df.high, df.low], axis = 1)
    featurevector.columns = ['date','RSI1', 'RSI2', 'variability', 'close', 'high', 'low']
    featurevector = featurevector.dropna()
    df = featurevector
    features = [ 'RSI1', 'RSI2', 'variability',  'high', 'low']
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    x = featurevector[features].copy()    
    y = featurevector['close'].copy()

    NUM_FEATURES = x.shape[1]
    
    x[features] = x_scaler.fit_transform(x)

    y = y_scaler.fit_transform(y.values.reshape(-1, 1))
    X_train, y_train = load.load_data(x, WINDOW, TrainTest = False)
    
    model = build_model(input_shape=(WINDOW, NUM_FEATURES))
    
    print('START FIT MODEL...')
    
    start = time.time()
    
    #history = History()
    #history= model.fit(X_train, y_train, validation_data=(X_test, y_test),  batch_size=32, epochs=100,verbose=1,
    #          callbacks=[history])
    
    model.fit(X_train, y_train, batch_size=32, epochs=500, verbose=1)
    end = time.time()

    print ('Learning time: ', end-start)
    
    today = time.strftime("_%d_%m_%Y")
    
    pathModel = "../../models/model_VarRSI_" + typeBlockchain + today +".h5"
    save_model(model, pathModel)
    
    #model = load_model(pathModel)
    # one day prediction. get last batch known data (now we didnt need in y value and can predict it)    
    lastbatch = np.array(x[-WINDOW:])
    pred = model.predict([lastbatch.reshape(1,22, NUM_FEATURES)])
    pred =  np.array(y_scaler.inverse_transform(pred)) # predicted value

    # now we make dataframe and create row names in date

    lastDate =str(df.date[df.last_valid_index()]).split('-')
    currentData = datetime.date(int(lastDate[0]),int(lastDate[1]),int(lastDate[2])) + datetime.timedelta(1)
    predictionDate = pd.date_range(currentData, periods=1)
    prediction = pd.DataFrame(pred, columns=["predictionPrice"], index = predictionDate.values)

    print (prediction)
    del model
    
    K.clear_session()


    
    
    return prediction