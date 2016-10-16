import talib
from pylab import *
import pandas as pd
import numpy as np

def __init__(self):
    self.n = 4

    try:
        con = sqlite3.connect("new_kospi.db")
        cursor = con.cursor()
        cursor.execute("SELECT * FROM new_kospi")
        rows = cursor.fetchall()

        self.k_date, self.jisu, self.highjisu, self.lowjisu = [], [], [], []

        for row in rows:
            self.k_date.append(int(row[0]))
            self.jisu.append(float(row[1]))
            self.highjisu.append(float(row[2]))
            self.lowjisu.append(float(row[3]))

        self.k_date.reverse()
        self.jisu.reverse()
        self.highjisu.reverse()
        self.lowjisu.reverse()
        self.arr_date = np.array(self.k_date)
        self.arr_close = np.array(self.jisu)
        self.arr_high = np.array(self.highjisu)
        self.arr_low = np.array(self.lowjisu)
    except:
        print("error database connection")


def RSI(self):
    rsi = talib.RSI(self.arr_close, timeperiod=4)
    return rsi

"""relative algorithm for RSI
from pylab import *
import pandas as pd
import numpy as np
def Datapull(Stock):
    try:
        df = (pd.io.data.DataReader(Stock, 'yahoo', start='01/01/2010'))
        return df
        print
        'Retrieved', Stock
        time.sleep(5)
    except Exception, e:
        print
        'Main Loop', str(e)
def RSIfun(price, n=14):
    delta = price['Close'].diff()
    # -----------
    dUp, dDown = delta.copy(), delta.copy()
    dUp[dUp < 0] = 0
    dDown[dDown > 0] = 0
    RolUp = pd.rolling_mean(dUp, n)
    RolDown = pd.rolling_mean(dDown, n).abs()
    RS = RolUp / RolDown
    rsi = 100.0 - (100.0 / (1.0 + RS))
    return rsi
Stock = 'AAPL'
df = Datapull(Stock)
RSIfun(df)
"""