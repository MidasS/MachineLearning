from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from pandas import Series, DataFrame
import pandas as pd
import pandas.io.data as web
import datetime
import tensorflow as tf
from sklearn import svm, preprocessing
import matplotlib.pyplot as plt
import math

class KOSPIDATA:

    def __init__(self):
        start = datetime.datetime(1998, 5, 1)
        end = datetime.datetime(2016, 12, 31)
        kospi = web.DataReader("^KS11", "yahoo", start, end)

        self.arr_date= np.array(kospi.index)
        self.arr_open = np.array(kospi.Open, dtype=float)
        self.arr_close= np.array(kospi['Adj Close'], dtype=float)
        self.arr_high= np.array(kospi.High, dtype=float)
        self.arr_low= np.array(kospi.Low, dtype=float)
        self.arr_volume= np.array(kospi.Volume, dtype=float)


if __name__ == "__main__":


    def create_dataset(dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-look_back-1):
            a = dataset[i:(i+look_back)]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)

    K = KOSPIDATA()

    # FEATURES = ['high', 'low', 'open', 'close', 'volume']
    FEATURES = ['high', 'low', 'open', 'close']

    data = {'year': K.arr_date,
            'open': K.arr_open,
            'high': K.arr_high,
            'low': K.arr_low,
            'close': K.arr_close}

    # df = DataFrame(data, columns=['year', 'high', 'low', 'open', 'close'])
    df = DataFrame(data, columns=['year','close'])





    train_size = int(len(df['close']) * 0.67)
    test_size = len(df['close']) - train_size
    train, test = df['close'].values[0:train_size], df['close'].values[train_size:len(df['close'].values)]
    # print(df['close'].values[0:10])

    look_back = 3
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    model = Sequential()
    model.add(Dense(8, input_dim=look_back, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, nb_epoch=200, batch_size=2, verbose=2)

    # Estimate model performance
    trainScore = model.evaluate(trainX, trainY, verbose=0)
    print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
    testScore = model.evaluate(testX, testY, verbose=0)
    print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))

    # generate predictions for training
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # shift train predictions for plotting
    trainPredictPlot = np.empty_like(df['close'])
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

    # shift test predictions for plotting
    testPredictPlot = np.empty_like(df['close'])
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict)+(look_back*2)+1:len(df['close'])-1, :] = testPredict

    # plot baseline and predictions
    plt.plot(df['close'])
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.show()





