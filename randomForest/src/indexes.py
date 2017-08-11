import numpy as np
import pandas as pd
from scipy import signal

def RSI(dataframe, period):
    '''
    Computes the RSI of a given price series for a given period length
    :param dataframe:
    :param period:
    :return dataframe with rsi:
    '''

    rsi = []

  
    all_prices = dataframe['close']
    diff = np.diff(all_prices) # length is 1 less than the all_prices
    for i in range(period):
        rsi.append(None) # because RSI can't be calculated until period prices have occured

    for i in range(len(diff) - period + 1):
        avg_gain = diff[i:period + i]
        avg_loss = diff[i:period + i]
        avg_gain = abs(sum(avg_gain[avg_gain >= 0]) / period)
        avg_loss = abs(sum(avg_loss[avg_loss < 0]) / period)
        if avg_loss == 0:
            rsi.append(100)
        elif avg_gain == 0:
            rsi.append(0)
        else:
            rs = avg_gain / avg_loss
            rsi.append(100 - (100 / (1 + rs)))

    dataframe['RSI'] = rsi
    return dataframe


def PROC(dataframe, period):
    '''
    Computes the PROC(price rate of change) of a given price series for a given period length
    :param dataframe:
    :param period:
    :return proc:
    '''

    proc = []

    all_prices = list(dataframe['close'])
    for i in range(period):
        proc.append(None) # because proc can't be calculated until period prices have occured
    for i in range(len(all_prices) - period):
        if len(all_prices) <= period:
            proc.append(None)
        else:
            calculated = (all_prices[i + period] - all_prices[i]) / all_prices[i]
            proc.append(calculated)
    dataframe['PROC'] = proc
    return dataframe


def SO(dataframe, period):
    
    so = []
    
    
    all_prices = list(dataframe['close'])

    for i in range(period):
        so.append(None)

    for i in range(len(all_prices) - period):
        C = all_prices[i]
        H = max(all_prices[i:i+period])
        L = min(all_prices[i:i+period])
        so.append(100 * ((C - L) / (H - L)))

    dataframe['SO'] = so
    return dataframe


def Williams_R(dataframe, period):
    '''
    Williams %R
    Calculates fancy shit for late usage. Nice!

    EXAMPLE USAGE:
    data = pandas.read_csv("./data/ALL.csv", sep=",",header=0,quotechar='"')
    wr = Williams_R(data)
    print(wr)

    '''
    
    wr = []
    
    
    all_prices = list(dataframe['close'])
    for i in range(period):
        wr.append(None) # because proc can't be calculated until period prices have occured

    for i in range(period-1,len(all_prices)-1):
        C = all_prices[i]
        H = max(all_prices[i-period+1:i])
        L = min(all_prices[i-period+1:i])
        wr_one = (
            ((H - C) 
             / (H - L)) * -100
        )
        if wr_one <=-100:
            wr.append(-100)
        elif wr_one >= 100:
            wr.append(100)
        else:
            wr.append(wr_one)
    dataframe["WR"] = wr
    return dataframe


def calculate_targets(df, period):
    
    targets = []

 
    all_prices = list(df['close'])

    for i in range(0, len(all_prices)-period):
        targets.append(np.sign(all_prices[i+period] - all_prices[i]))
    for i in range(len(all_prices)-period, len(all_prices)):
        targets.append(None)

    df["Target({})".format(period)] = targets
    return df


def On_Balance_Volume(dataframe):
    '''
    Williams %R
    Calculates fancy shit for late usage. Nice!

    EXAMPLE USAGE:
    data = pandas.read_csv("./data/ALL.csv", sep=",",header=0,quotechar='"')
    wr = Williams_R(data)
    print(wr)

    '''
    obv = []
    
   
    all_prices = list(dataframe['close'])
    all_volumes = list(dataframe['volume'])

    obv.append(dataframe.iloc[0]["volume"])
    for i in range(1,len(all_prices)):
        C_old = all_prices[i-1]
        C = all_prices[i]
        if(C > C_old):
            obv.append(obv[i-1]+ all_volumes[i])
        elif (C < C_old):
            obv.append(obv[i - 1] - all_volumes[i])
        else:
            obv.append(obv[i-1])

    dataframe['OBV'] = obv
    return dataframe


def detrend_data(df):
    trend = None
    all_prices = list(df['close'])
#        trend.append(signal.detrend(all_prices))
    if(trend is None):
        trend = list(signal.detrend(all_prices))
    else:
        trend.extend(signal.detrend(all_prices))

    df['Close_detrend'] = trend
    return df