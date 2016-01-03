import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, preprocessing
import pandas as pd
from matplotlib import style
style.use("ggplot")



FEATURES = ['width','after_T[-6]','after_T[-5]','after_T[-4]','after_T[-3]','after_T[-2]','after_T[-1]','after_T[0]','after_T[1]','after_T[2]','after_T[3]','after_T[4]','after_T[5]','after_T[6]', 'before_T[-6]', 'before_T[-5]', 'before_T[-4]', 'before_T[-3]', 'before_T[-2]', 'before_T[-1]', 'before_T[0]', 'before_T[1]', 'before_T[2]', 'before_T[3]', 'before_T[4]', 'before_T[5]', 'before_T[6]']




def Build_Data_Set():
    data_df = pd.DataFrame.from_csv("ng_data.csv")
    data_df = data_df.reindex(np.random.permutation(data_df.index))
    # print(data_df)

    X = np.array(data_df[FEATURES].values)
    X = preprocessing.scale(X)
    y = (data_df["result_2"].values.tolist())

    data_df2 = pd.DataFrame.from_csv("ng_data2.csv")
    data_df2 = data_df2.reindex(np.random.permutation(data_df2.index))

    X2 = np.array(data_df2[FEATURES].values)
    X2 = preprocessing.scale(X2)
    y2 = (data_df2["result_2"].values.tolist())


    return X,y,X2,y2

def Analysis():
    X, y = Build_Data_Set()

    clf = svm.SVC(kernel="linear", C= 1.0)
    clf.fit(X,y)
    # print(clf.predict([[8,20]]))
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(min(X[:, 0]), max(X[:, 0]))
    yy = a * xx - clf.intercept_[0] / w[1]
    # print(xx,yy)
    h0 = plt.plot(xx,yy, "k-", label="non weighted")

    plt.scatter(X[:, 0],X[:, 1],c=y)
    plt.ylabel("avg_T")
    plt.xlabel("width")
    plt.legend()

    plt.show()

# Analysis()

def Analysis2():

    test_size = 1000
    X, y,X2,y2 = Build_Data_Set()
    print(len(X2[:-test_size]))

    # print(X[:])
    clf = svm.SVC(kernel="linear", C= 1.0)
    clf.fit(X[:-test_size],y[:-test_size])
    # print(clf.fit(X[:-test_size],y[:-test_size]))

    correct_count = 0


    for x in range(1, test_size+1):
        # print(len(clf.predict(X[-x])))
        if clf.predict(X[-x])[0] == y[-x]:
            correct_count += 1

    print("Accuracy:", (correct_count/test_size) * 100.0000)

def Analysis3():

    rr2 =[]
    X, y,X2,y2 = Build_Data_Set()
    # print(len(X[:-test_size]))


    # print(X[:])
    clf = svm.SVC(kernel="linear", C= 10.0)
    clf.fit(X[:],y[:])
    # print(clf.fit(X[:-test_size],y[:-test_size]))
    correct_count = 0
    # for x in range  (1,len(X2)):
    #     rr2.append(clf.predict(X2[x])[0])
    # print(len(y2))
    # print(rr2.count(1))
    for x in range(1, len(X2[:])):
        if clf.predict(X2[x])[0] == y2[x] ==0:
            correct_count += 1

    # print("Accuracy:", (correct_count/len(y2)) * 100.00)

    print(correct_count)
    print(y2.count(0))
    print("Accuracy:", (correct_count/y2.count(0)) * 100.00)




Analysis3()