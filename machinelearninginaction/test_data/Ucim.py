__author__ = 'blue1'
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, preprocessing
import pandas as pd
from matplotlib import style
style.use("ggplot")

def Build_Data_Set(features = ["width",
                               "average_T"]):
    data_df = pd.DataFrame.from_csv("ng_data.csv")
    print(data_df)

    X = np.array(data_df[features].values)
    X = preprocessing.scale(X)
    print(X)
    y = (data_df["result_2"].values.tolist())
    print(y)

    return X,y

def Analysis():
    X, y = Build_Data_Set()

    clf = svm.SVC(kernel="linear", C= 1.0)
    clf.fit(X,y)

    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(min(X[:, 0]), max(X[:, 0]))
    yy = a * xx - clf.intercept_[0] / w[1]
    print(xx,yy)
    h0 = plt.plot(xx,yy, "k-", label="non weighted")

    plt.scatter(X[:, 0],X[:, 1],c=y)
    plt.ylabel("average_T")
    plt.xlabel("width")
    plt.legend()

    plt.show()

Analysis()
