from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from pandas import Series, DataFrame
import pandas as pd
import pandas.io.data as web
import datetime
import tensorflow as tf
from sklearn import svm, preprocessing


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

    K = KOSPIDATA()

    # FEATURES = ['high', 'low', 'open', 'close', 'volume']
    FEATURES = ['high', 'low', 'open', 'close']

    data = {'year': K.arr_date,
            'open': K.arr_open,
            'high': K.arr_high,
            'low': K.arr_low,
            'close': K.arr_close}

    df = DataFrame(data, columns=['year', 'high', 'low', 'open', 'close'])
    # print(df[df.volume==0].index)

    # df = df.drop([ '484',  '823',  '828',  829,  830,  831,  832,  833,  956, 3887, 3927, 3945, 4097, 4101, 4105, 4109, 4112, 4154, 4155, 4205, 4427, 4614])
    # df = df.drop(df.ix[484], axis=1)
    # print(df.ix[484])


    def prediction():
        profit = []
        for i in range(len(df['close'])-1):
            if df['close'][i] < df['close'][i+1]:
                profit.append([1,0])
            else :
                profit.append([0,1])
        profit.append([2,2])

        return profit

    print(df)


    profit = np.array(prediction())
    # profit = np.append(profit,np.NaN)

    print(profit.shape)

    # df['profit'] = profit




    new_XX = np.array(df[FEATURES].values[:-2, :])
    print(new_XX.shape)
    print(new_XX)
    # new_YY = df['profit'].values[:-2]
    new_YY = profit[:-2]
    # print(new_Y)
    # new_YY = np.reshape(new_Y, [4616,2])
    print(new_YY)


    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=4, init='uniform', activation='relu'))
    model.add(Dense(8, init='uniform', activation='relu'))
    model.add(Dense(6, init='uniform', activation='relu'))
    model.add(Dense(2, init='uniform', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy' , optimizer='adam', metrics=['accuracy'])
    # Fit the model
    model.fit(new_XX, new_YY, nb_epoch=150, batch_size=20)
    # evaluate the model
    scores = model.evaluate(new_XX, new_YY)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))