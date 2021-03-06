from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from pandas import Series, DataFrame
import pandas as pd
import pandas.io.data as web
import datetime
import tensorflow as tf
from sklearn import svm, preprocessing
from keras.layers.normalization import BatchNormalization
import talib


class KOSPIDATA:
    def __init__(self):
        start = datetime.datetime(1998, 5, 1)
        end = datetime.datetime(2016, 12, 31)
        kospi = web.DataReader("^KS11", "yahoo", start, end)

        self.arr_date = np.array(kospi.index)
        self.arr_open = np.array(kospi.Open, dtype=float)
        self.arr_close = np.array(kospi['Adj Close'], dtype=float)
        self.arr_high = np.array(kospi.High, dtype=float)
        self.arr_low = np.array(kospi.Low, dtype=float)
        self.arr_volume = np.array(kospi.Volume, dtype=float)


    def MA5(self):
        ma5 = talib.MA(self.arr_close, timeperiod=5)
        return ma5

    def MA10(self):
        ma10 = talib.MA(self.arr_close, timeperiod=10)
        return ma10

    def StoK(self):
        fastk, fastd= talib.STOCHF(self.arr_high, self.arr_low,self.arr_close, fastk_period=15, fastd_matype=0)
        slowk, slowd = talib.STOCH(self.arr_high, self.arr_low, self.arr_close, fastk_period= 15, slowk_period=5, slowk_matype=0, slowd_period=3, slowd_matype=0)
        return fastk, fastd, slowd

    def Momentum(self):
        momentum = self.arr_close - np.roll(self.arr_close,4)
        return momentum

    def ROC(self):
        roc = (self.arr_close / np.roll(self.arr_close,4)) * 100
        return roc


    def AD_Oscil(self):
        ad_oscil = (self.arr_high - np.roll(self.arr_close,1))/(self.arr_high - self.arr_low)
        return ad_oscil

    def DIsp5(self):
        disp5 = (self.arr_close/talib.MA(self.arr_close, timeperiod =5))*100
        return disp5

    def DIsp10(self):
        disp10 = (self.arr_close/talib.MA(self.arr_close, timeperiod =10))*100
        return disp10

    def OSCP(self):
        oscp = (self.MA5()-self.MA10())/self.MA5()
        return oscp

    def CCI(self):
        cci = talib.CCI(self.arr_high, self.arr_low, self.arr_close, timeperiod=7)
        return cci

    def RSI(self):
        rsi = talib.RSI(self.arr_close, timeperiod =4)
        return rsi

if __name__ == "__main__":

    K = KOSPIDATA()

    # FEATURES = ['high', 'low', 'open', 'close', 'volume']
    FEATURES = ['high', 'low', 'open', 'close','ma5', 'ma10', 'fastk', 'fastd', 'slowd','momentum','roc','ad_oscil','disp5','disp10', 'oscp','cci','rsi']

    data = {'year': K.arr_date,
            'open': K.arr_open,
            'high': K.arr_high,
            'low': K.arr_low,
            'close': K.arr_close}

    df = DataFrame(data, columns=['year', 'high', 'low', 'open', 'close','ma5', 'ma10', 'fastk', 'fastd', 'slowd','momentum','roc','ad_oscil','disp5','disp10', 'oscp','cci','rsi'])


    # print(df[df.volume==0].index)

    # df = df.drop([ '484',  '823',  '828',  829,  830,  831,  832,  833,  956, 3887, 3927, 3945, 4097, 4101, 4105, 4109, 4112, 4154, 4155, 4205, 4427, 4614])
    # df = df.drop(df.ix[484], axis=1)
    # print(df.ix[484])




    df.fastk, df.fastd, df.slowd = K.StoK()
    df.ad_oscil = K.AD_Oscil()
    df.ad_oscil[0] = np.NaN
    df.momentum = K.Momentum()
    df.momentum[0:4] = np.NaN
    df.roc = K.ROC()
    df.ma5, df.ma10 ,df.disp5, df.disp10 = K.MA5(), K.MA10(), K.DIsp5(), K.DIsp10()
    df.oscp , df.cci = K.OSCP(), K.CCI()
    df.rsi = K.RSI()


    def prediction():
        profit = []
        for i in range(len(df['close']) - 1):
            if df['close'][i] < df['close'][i + 1]:
                profit.append([1, 0])
            else:
                profit.append([0, 1])
        profit.append([2, 2])

        return profit


    print(df)

    profit = np.array(prediction())
    # profit = np.append(profit,np.NaN)

    print(profit.shape)

    # df['profit'] = profit

    new_XX = np.array(df[FEATURES].values[20:-2, :])
    print(new_XX.shape)
    print(new_XX)
    # new_YY = df['profit'].values[:-2]
    new_YY = profit[20:-2]
    # print(new_Y)
    # new_YY = np.reshape(new_Y, [4616,2])
    print(new_YY)

    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=17, init='uniform', activation='relu'))
    # model.add(BatchNormalization())
    model.add(Dense(8, init='uniform', activation='relu'))
    # model.add(BatchNormalization())
    model.add(Dense(6, init='uniform', activation='relu'))
    # model.add(BatchNormalization())
    model.add(Dense(2, init='uniform', activation='sigmoid'))
    # model.add(BatchNormalization())



    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Fit the model
    model.fit(new_XX, new_YY, validation_split=0.25, nb_epoch=150, batch_size=10)
    # evaluate the model
    scores = model.evaluate(new_XX, new_YY)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))