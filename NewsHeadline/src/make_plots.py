import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.offline import iplot
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import numpy as np
import pandas as pd

WINDOW = 22

def plotPlotly_N_feature2(feature_1,feature_2,feature_3, label_1, label_2, label_3, indexes):
    feature_1  =   go.Scatter(x = indexes, y = feature_1,name=label_1)
    feature_2  =   go.Scatter( x = indexes, y = feature_2,name=label_2)
    feature_3 =   go.Scatter( x = indexes, y = feature_3,name=label_3)

    iplot([feature_1,feature_2,feature_3])
    plt.show()
    
def plotAsScaler2(feature_1, feature_2, feature_3, indexes):

    f_1_scaler = MinMaxScaler()
    f_2_scaler = MinMaxScaler()
    f_3_scaler = MinMaxScaler()
    
    feature_1 = f_1_scaler.fit_transform(feature_1)
    feature_2 = f_2_scaler.fit_transform(feature_2)
    feature_3 = f_3_scaler.fit_transform(feature_3)

    plotPlotly_N_feature(feature_1,feature_2,feature_3,'neg', 'neu', 'pos', indexes)




def plotPlotly_N_feature(feature_1, feature_2, feature_3, label_1, label_2, label_3):
    feature_1 = go.Scatter(y=feature_1, name=label_1)
    feature_2 = go.Scatter(y=feature_2, name=label_2)
    feature_3 = go.Scatter(y=feature_3, name=label_3)

    iplot([feature_1, feature_2, feature_3])
    plt.show()


def plotAsScaler(feature_1, feature_2, feature_3):
    f_1_scaler = MinMaxScaler()
    f_2_scaler = MinMaxScaler()
    f_3_scaler = MinMaxScaler()

    feature_1 = f_1_scaler.fit_transform(feature_1.close)
    feature_2 = f_2_scaler.fit_transform(feature_2.close)
    feature_3 = f_3_scaler.fit_transform(feature_3.close)

    plotPlotly_N_feature(feature_1, feature_2, feature_3, 'USDT_BTC', 'USDT_ETH', 'USDT_DASH')


def plot_quality_graph(dataframe, model, X_train, X_test, y_train, \
                       y_test, y_scaler):

    trainPredict = model.predict(X_train)
    testPredict = model.predict(X_test)

    trainPredict = y_scaler.inverse_transform(trainPredict)
    trainY = y_scaler.inverse_transform([y_train])

    testPredict = y_scaler.inverse_transform(testPredict)
    testY = y_scaler.inverse_transform([y_test])

    trainScore = metrics.mean_squared_error(trainY[0], trainPredict[:, 0]) ** .5
    print('Train Score: %.2f RMSE' % (trainScore))

    testScore = metrics.mean_squared_error(testY[0], testPredict[:, 0]) ** .5
    print('Test Score: %.2f RMSE' % (testScore))

    prices = dataframe.close.values.astype('float32')
    prices = prices.reshape(len(prices), 1)
    trainPredictPlot = np.empty_like(prices)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[WINDOW:len(trainPredict) + WINDOW, :] = trainPredict

    testPredictPlot = np.empty_like(prices)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[(len(prices) - testPredict.shape[0]):len(prices), :] = testPredict

    plt.plot(pd.DataFrame(prices, columns=["close"], index=dataframe.index).close, label='Actual')
    plt.plot(pd.DataFrame(trainPredictPlot, columns=["close"], index=dataframe.index).close, label='Training')
    plt.plot(pd.DataFrame(testPredictPlot, columns=["close"], index=dataframe.index).close, label='Testing')
    plt.legend(loc='best')
    plt.show()
    return prices, trainPredictPlot, testPredictPlot, testPredict, testY


def plot_interactive(dataframe, prices, trainPredictPlot, testPredictPlot):
    Actual = pd.DataFrame(prices, columns=["close"], index=dataframe.index).close
    Training = pd.DataFrame(trainPredictPlot, columns=["close"], index=dataframe.index).close
    Testing = pd.DataFrame(testPredictPlot, columns=["close"], index=dataframe.index).close

    ActualValues = go.Scatter(x=dataframe.index, y=Actual, name='ActualValues')
    TrainingValues = go.Scatter(x=dataframe.index, y=Training, name='TrainingValues')
    TestingValues = go.Scatter(x=dataframe.index, y=Testing, name='PredictedValues')

    iplot([ActualValues, TrainingValues, TestingValues])
    plt.show()


def plotHictory(history):
    plt.plot(history.history['loss'], label='TrainLoss')
    plt.plot(history.history['val_loss'], label='TestLoss')
    plt.legend()
    plt.show()