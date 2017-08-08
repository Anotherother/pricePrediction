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

############## МНОГО РАЗНЫХ ИНДИКАТОРОВ ########

def wilder_smooth(value_list, period):
    nan_offset = np.isnan(value_list).sum()
    smoothened = [np.nan for i in range(period-1+nan_offset)]
    smoothened.append(np.mean(value_list[nan_offset:period+nan_offset]))
    for i in range(period+nan_offset, len(value_list)):
        smoothened.append(smoothened[i-1] + (value_list[i] - smoothened[i-1]) / period)
    return np.array(smoothened)

def ma_rel_diff(df, period=50):
    df['ma_rel_diff_{}'.format(period)] = 1 - df['close'].rolling(period).mean().values / df['close']
    return df

def ema_rel_diff(df, period=10):
    df['ema_rel_diff_{}'.format(period)] = 1 - df['close'].ewm(span=period, min_periods=period-1).mean().values / df['close']
    return df

def mom(df, period=20):
    df['moment_{}'.format(period)] = df['close'].diff(period).values
    return df

def roc(df, period=14):
    df['roc_{}'.format(period)] = df['close'] / df['close'].shift(period).values
    return df

def bbands(df, period=20, std_off=2):
    boil_mean = df['close'].rolling(period).mean().to_frame(name='boil_mean_{}_{}'.format(period, std_off))
    boil_std = df['close'].rolling(period).std().to_frame(name='boil_std_{}_{}'.format(period, std_off))

    boil_up = (boil_mean['boil_mean_{}_{}'.format(period, std_off)] + std_off*boil_std['boil_std_{}_{}'.format(period, std_off)]).to_frame(name='boil_up_{}_{}'.format(period, std_off))
    boil_down = (boil_mean['boil_mean_{}_{}'.format(period, std_off)] - std_off*boil_std['boil_std_{}_{}'.format(period, std_off)]).to_frame(name='boil_down_{}_{}'.format(period, std_off))

    df = df.join(boil_mean)
    df = df.join(boil_up)
    df = df.join(boil_down)

    return df

def normalized_bbands(df, period=20, std_off=20):
    boil_mean = df['close'].rolling(period).mean()
    boil_std = df['close'].rolling(period).std()

    boil_up = df['close'].values / (boil_mean + std_off*boil_std) - 1
    boil_down = df['close'].values / (boil_mean - std_off*boil_std) - 1

    boil_up = boil_up * boil_up.gt(0)
    boil_down = boil_down * boil_down.lt(0)

    df['normBB'] = boil_up.values + boil_down.values

    return df


def rsi(df, period=14):
    df['rsi_{}'.format(period)] = 100.0 - 100.0 / (1.0 + df['close'].diff(1).gt(0).rolling(period).mean().values / df['close'].diff(1).lt(0).rolling(period).mean().values)
    return df


def stochastics(df, period=14, smooth=3):
    stoch_k = (100.0 * (df['close'] - df['low'].rolling(period).min()) / (df['high'].rolling(period).max() - df['low'].rolling(period).min())).to_frame(name='stoch_k_{}'.format(period))
    stoch_d = stoch_k['stoch_k_{}'.format(period)].rolling(smooth).mean().to_frame(name='stoch_d_{}_{}'.format(period, smooth))
    df = df.join(stoch_d)

    return df


def macd(df, period_fast=12, period_slow=26, smooth=9):
    macd = df['close'].ewm(span=period_fast, min_periods=period_fast-1).mean() - df['close'].ewm(span=period_slow, min_periods=period_slow-1).mean()
    macd_smoothed = macd.ewm(span=smooth, min_periods=smooth-1).mean().values

    df['macd_{}_{}_{}'.format(period_fast, period_slow, smooth)] = macd - macd_smoothed

    return df


def atr(df, period=14):
    TR = pd.concat([df['high'], df['close'].shift(1)], 1).max(1) - pd.concat([df['low'], df['close'].shift(1)], 1).min(1)
    ATR = TR.rolling(period).mean().to_frame(name='atr_{}'.format(period))
    return df.join(ATR)


def adx(df, period=14):
    TR = pd.concat([df['high'], df['close'].shift(1)], 1).max(1) - pd.concat([df['low'], df['close'].shift(1)], 1).min(1)
    df['ATR'] = wilder_smooth(TR, period)

    up_down = df['high'].diff(1).gt(-1*df['low'].diff(1))

    pDM = df['high'].diff(1) * df['high'].diff(1).gt(0) * up_down
    mDM = df['low'].diff(1) * df['low'].diff(1).lt(0) * (up_down - 1)

    pDI = 100 * wilder_smooth(pDM.values, period) / df['ATR']
    mDI = 100 * wilder_smooth(mDM.values, period) / df['ATR']

    DX = (100 * (pDI - mDI).abs() / (pDI + mDI))

    ADX = pd.DataFrame(wilder_smooth(DX, period), index=df.index, columns=['adx_{}'.format(period)])

    return df.join(ADX)


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
    
    plot = True
    plotHictory = False
    interactiveGrapth = True
    plotForTrain = False
    
    df = get_data.get_data_frame(typeBlockchain, stock)    
    df.index  = df.date
    df = df[[ 'open','close','low','high','volume']]

    df = ma_rel_diff(df)
    df = ema_rel_diff(df)
    df = mom(df)
    df = roc(df)
    df = bbands(df)
    df = normalized_bbands(df)
    df = rsi(df)
    df = stochastics(df)
    df = macd(df)
    df = atr(df)
    df = adx(df)
    df = df.dropna()
    
    
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    all_df = df.copy()
    
    features = ['macd_12_26_9', 'stoch_d_14_3', 'roc_14']
 #             ['moment_20', 'ema_rel_diff_10', 'ma_rel_diff_50'],\
 #             ['atr_14', 'moment_20'],
 #             ['atr_14', 'moment_20','low', 'high'],
 #             ['roc_14', 'moment_20', 'ema_rel_diff_10' ], 
 #             ['roc_14', 'rsi_14'], ['roc_14', 'rsi_14', 'macd_12_26_9']
             

    x = all_df[features].copy()

    y = all_df['close'].copy()
    NUM_FEATURES = x.shape[1]

    x[features] = x_scaler.fit_transform(x)


    y = y_scaler.fit_transform(y.values.reshape(-1, 1))        
    x['close'] = y ####!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    X_train, y_train, X_test, y_test = load.load_data(x, WINDOW, train_size= 0.96, TrainTest = True)

    x = all_df[features].copy() ####!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    model = build_model(input_shape=(WINDOW, NUM_FEATURES))

    print('START FIT MODEL...')
    print(features)
    print()
    start = time.time()

    #history = History()
    #history= model.fit(X_train, y_train, validation_data=(X_test, y_test),  batch_size=32,\
    #                   epochs=500,verbose=0,
    #          callbacks=[history])

    model.fit(X_train, y_train, batch_size=32, epochs=500, verbose=1)
    end = time.time()

    print ('Learning time: ', end-start)

    today = time.strftime("_%d_%m_%Y")
    pathModel = "./model_" + str(features) + typeBlockchain + today +".h5"
    #pathModel = "../../models/model_low_high_USDT_BTC_03_08_2017.h5"
    save_model(model, pathModel)

    #model = load_model(pathModel)
    lastbatch = np.array(x[-WINDOW:])
    pred = model.predict([lastbatch.reshape(1,WINDOW, NUM_FEATURES)])
    pred =  np.array(y_scaler.inverse_transform(pred)) # predicted value

    # one day prediction. get last batch known data (now we didnt need in y value and can predict it)    
    lastDate =str(df.last_valid_index()).split('-')
    currentData = datetime.date(int(lastDate[0]),int(lastDate[1]),int(lastDate[2])) + datetime.timedelta(1)
    predictionDate = pd.date_range(currentData, periods=1)
    prediction = pd.DataFrame(pred, columns=["predictionPrice"], index = predictionDate.values)


    print (prediction)
    return prediction