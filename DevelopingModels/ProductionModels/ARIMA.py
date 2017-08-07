import src.get_data as get_data
import src.load_data as load

import pandas as pd
import numpy as np


#[1] THEORY: https://people.duke.edu/~rnau/411arim3.htm
#[2] Library: ARIMA http://www.pyflux.com/docs/arima.html
#[3] OPTIMISATION AR, MA: http://thingsolver.com/forecasting-with-arima/

from statsmodels.tsa.arima_model import ARIMA

def oldARIMA(currency, numsteps):
    """
        this model paint one line, which apriximate trend in one line
        Linear model
    """
    df = get_data.get_data_frame(currency)

    ts = df[['date','close']][:-10]

    ts_log = np.log(ts.close)
    model = ARIMA(ts_log.dropna().as_matrix(), order=(1, 1, 0))  
    results_ARIMA = model.fit()  
    stepahead = results_ARIMA.forecast(numsteps)[0]
    prediction = np.exp(stepahead)
    print ('%s Steps Ahead Forecast  is:' % numsteps, prediction) 
    return np.exp(stepahead)

oldARIMA("USDT_BTC", 3)

##########################################
import pyflux as pf

def newARIMA(currency, numsteps):

    df = get_data.get_data_frame(currency)
    ts_log = np.log(df.close[:-2])
    
    model= pf.ARIMA(data=ts_log.as_matrix(), ar=1, ma=2, integ=0) # AR and MA compute not as an example = experiments
    x_mle = model.fit("MLE") #maximum likelihood estimate
    
    y_pred = np.exp(model.predict(numsteps))
    
    print (y_pred)
       
    return y_pred

newARIMA("USDT_BTC", 3)