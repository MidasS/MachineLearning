import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, preprocessing
import pandas as pd
from matplotlib import style
import math


df = pd.DataFrame.from_csv("TimeTablerf.csv")
for x in df.columns:
    print(x)

print()
Bdf = df['Birthrate'].values
Ddf = df['Deathrate'].values
# print(type(Bdf))
#
#
#
# print(int(Bdf))


def pearson(x,y):
    n=len(x)
    vals=range(n)
    # 합한다.
    sumx = sum([float(x[i]) for i in vals])
    sumy = sum([float(y[i]) for i in vals])

    # 제곱을 합한다.
    sumxSq = sum([x[i]**2.0 for i in vals])
    sumySq = sum([y[i]**2.0 for i in vals])

    # 곱을 합한다.
    pSum = sum([x[i]*y[i] for i in vals])

    # 피어슨 점수를 계산한다.
    num = pSum - (sumx*sumy/n)
    den = ((sumxSq-pow(sumx,2)/n)*(sumySq-pow(sumy,2)/n))**.5
    if den==0: return 0

    r = num/den

    return r

print(pearson(Bdf,Ddf))