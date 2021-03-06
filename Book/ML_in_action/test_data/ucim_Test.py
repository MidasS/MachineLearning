import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, preprocessing
import pandas as pd
from matplotlib import style
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
style.use("ggplot")





class Data_Set():

    # FEATURES = ['width','avg_T','std_T','rel_T[-6]','rel_T[-5]','rel_T[-4]','rel_T[-3]','rel_T[-2]','rel_T[-1]','rel_T[0]','rel_T[1]','rel_T[2]','rel_T[3]','rel_T[4]','rel_T[5]','rel_T[6]','after_T[-6]','after_T[-5]','after_T[-4]','after_T[-3]','after_T[-2]','after_T[-1]','after_T[0]','after_T[1]','after_T[2]','after_T[3]','after_T[4]','after_T[5]','after_T[6]', 'before_T[-6]', 'before_T[-5]', 'before_T[-4]', 'before_T[-3]', 'before_T[-2]', 'before_T[-1]', 'before_T[0]', 'before_T[1]', 'before_T[2]', 'before_T[3]', 'before_T[4]', 'before_T[5]', 'before_T[6]']
    # FEATURES = ['width','avg_T','std_T','rel_T[-3]','rel_T[-2]','rel_T[-1]','rel_T[0]','rel_T[1]','rel_T[2]','rel_T[3]','after_T[-3]','after_T[-2]','after_T[-1]','after_T[0]','after_T[1]','after_T[2]','after_T[3]']
    # FEATURES = ['width','avg_T','std_T','rel_T[-6]','rel_T[-5]','rel_T[-4]','rel_T[-3]','rel_T[-2]','rel_T[-1]','rel_T[0]','rel_T[1]','rel_T[2]','rel_T[3]','rel_T[4]','rel_T[5]','rel_T[6]']
    # FEATURES = ['width','avg_T','std_T','after_T[-6]','after_T[-5]','after_T[-4]','after_T[-3]','after_T[-2]','after_T[-1]','after_T[0]','after_T[1]','after_T[2]','after_T[3]','after_T[4]','after_T[5]','after_T[6]', 'before_T[-6]', 'before_T[-5]', 'before_T[-4]', 'before_T[-3]', 'before_T[-2]', 'before_T[-1]', 'before_T[0]', 'before_T[1]', 'before_T[2]', 'before_T[3]', 'before_T[4]', 'before_T[5]', 'before_T[6]']
    # FEATURES = ['width','avg_T','std_T','rel_T[-3]','rel_T[-2]','rel_T[-1]','rel_T[0]','rel_T[1]','rel_T[2]','rel_T[3]']
    # FEATURES = ['width','avg_T','std_T','after_T[-3]','after_T[-2]','after_T[-1]','after_T[0]','after_T[1]','after_T[2]','after_T[3]']

    # FEATURES = ['after_T[-6]','after_T[-5]','after_T[-4]','after_T[-3]','after_T[-2]','after_T[-1]','after_T[0]','after_T[1]','after_T[2]','after_T[3]','after_T[4]','after_T[5]','after_T[6]', 'before_T[-6]', 'before_T[-5]', 'before_T[-4]', 'before_T[-3]', 'before_T[-2]', 'before_T[-1]', 'before_T[0]', 'before_T[1]', 'before_T[2]', 'before_T[3]', 'before_T[4]', 'before_T[5]', 'before_T[6]']
    # FEATURES = ['avg_T','std_T','width','rel_max_T','rel_min_T','rel_T[-6]','rel_T[-5]','rel_T[-4]','rel_T[-3]','rel_T[-2]','rel_T[-1]','rel_T[0]','rel_T[1]','rel_T[2]','rel_T[3]','rel_T[4]','rel_T[5]','rel_T[6]']
    # FEATURES = ['avg_T','std_T', 'rel_max_T','rel_min_T','rel_T[-6]','rel_T[-5]','rel_T[-4]','rel_T[-3]','rel_T[-2]','rel_T[-1]','rel_T[0]','rel_T[1]','rel_T[2]','rel_T[3]','rel_T[4]','rel_T[5]','rel_T[6]','after_T[-6]','after_T[-5]','after_T[-4]','after_T[-3]','after_T[-2]','after_T[-1]','after_T[0]','after_T[1]','after_T[2]','after_T[3]','after_T[4]','after_T[5]','after_T[6]', 'before_T[-6]', 'before_T[-5]', 'before_T[-4]', 'before_T[-3]', 'before_T[-2]', 'before_T[-1]', 'before_T[0]', 'before_T[1]', 'before_T[2]', 'before_T[3]', 'before_T[4]', 'before_T[5]', 'before_T[6]']
    FEATURES = ['avg_T','std_T','rel_min_T','after_T[-6]','rel_T[-5]','rel_T[-4]','rel_T[-3]','rel_T[-2]','rel_T[0]','rel_T[1]','rel_T[2]','rel_T[3]','rel_T[4]','rel_T[5]','rel_T[6]']



    def Build_Data_Set(self):
        data_df = pd.DataFrame.from_csv("Final_Training.csv")
        data_df = data_df.reindex(np.random.permutation(data_df.index))

        self.X = np.array(data_df[self.FEATURES].values)
        self.X = preprocessing.scale(self.X)
        self.y = (data_df["result_2"].values.tolist())

        data_df2 = pd.DataFrame.from_csv("Final_Testing.csv")
        data_df2 = data_df2.reindex(np.random.permutation(data_df2.index))

        self.X2 = np.array(data_df2[self.FEATURES].values)
        self.X2 = preprocessing.scale(self.X2)
        self.y2 = (data_df2["result_2"].values.tolist())


        return self.X,self.y

def Corr():
    data_df = pd.DataFrame.from_csv("trim_data.csv")
    # print(data_df.corr(method='pearson'))
    corr_df = data_df.corr(method='pearson')
    print(corr_df>0.9)

def Analysis():
    data_1 = Data_Set()

    X, y = data_1.Build_Data_Set()

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

def Analysis2_2():
    data_1 = Data_Set()
    X, y = data_1.Build_Data_Set()
    print(X.shape)
    print(X)
    c_scores = []

    clf = svm.SVC(kernel="rbf", C= 10, gamma= 0.003)
    scores = cross_val_score(clf,X,y, cv=5, scoring='accuracy')
    c_scores.append(scores.mean())

    print(c_scores)



def Analysis2():
    data_1 = Data_Set()
    X, y = data_1.Build_Data_Set()

    c_range = np.arange(1,1000,10)
    # g_range = np.arange(0.001,0.05,0.002)
    c_scores = []
    c2_scores = []
    c3_scores = []
    print(c_range)
    for c in c_range:
        clf = svm.SVC(kernel="rbf", C= c, gamma= 0.005)
        scores = cross_val_score(clf,X,y, cv=5, scoring='accuracy')
        c_scores.append(scores.mean())

        clf2 = svm.SVC(kernel="poly", C= c, gamma= 0.005)
        scores2 = cross_val_score(clf2,X,y, cv=5, scoring='accuracy')
        c2_scores.append(scores2.mean())

        clf3 = svm.SVC(kernel="sigmoid", C= c, gamma= 0.005)
        scores3 = cross_val_score(clf3,X,y, cv=5, scoring='accuracy')
        c3_scores.append(scores3.mean())

    # dic_1 = {c_range :c_scores, c_range : c2_scores, c_range :c3_scores}
    # print(dic_1)
    print(c_scores)

    plt.plot(c_range,c_scores, 'r-', c_range, c2_scores, 'b--', c_range, c3_scores, 'g:')
    plt.xlabel('Value of C for SVM')
    plt.ylabel('Cross-Validated Accuracy')
    plt.show()


def Analysis2_5():
    data_1 = Data_Set()
    X, y = data_1.Build_Data_Set()

    k_list = ['rbf']
    # k_list = ['poly']
    # k_list = ['sigmoid']
    c_range = [0.1, 1, 10, 30, 50, 70, 100]
    g_range = [0.005, 0.003, 0.001, 0.0005, 0.0001, 0.00001]

    parameters = dict(kernel= k_list ,C= c_range, gamma=g_range )
    # print(parameters)

    grid = GridSearchCV(svm.SVC(),parameters, cv=5)
    grid.fit(X,y)

    clf = grid.best_estimator_
    clf2 = grid.best_score_
    clf3 = grid.param_grid
    # print(grid.grid_scores_)
    print(clf)
    print(clf2)
    print(clf3)


    # grid = GridSearchCV(svm.SVC, param_grid, cv=10, scoring='accuracy')
    # grid.fit(X,y)
    #
    # print(grid.best_score_)
    # print(grid.best_params_)
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm

def Analysis2_6():

    data_1 = Data_Set()
    X, y = data_1.Build_Data_Set()


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # c_range = np.arange(1,100,2)
    # g_range = np.arange(0.0001,0.01,0.0005)


    c_range = [0.1,0.5, 1, 10, 30, 50, 70, 100]
    g_range = [0.005, 0.003, 0.001, 0.0005, 0.0001, 0.00001]

    c_range2=[]
    g_range2=[]
    c_scores =[]

    for c in c_range:
        for g in g_range:
            clf = svm.SVC(kernel="sigmoid", C= c, gamma= g)
            scores = cross_val_score(clf,X,y, cv=5, scoring='accuracy')
            c_range2.append(c)
            g_range2.append(g)
            c_scores.append(scores.mean())

    ax.scatter(c_range2, g_range2, c_scores, c='r', marker='o')

    ax.set_axis_bgcolor('white')
    # ax.set_xlabel('Value of C')
    # ax.set_ylabel('Value of Gamma')
    # ax.set_zlabel('Accuracy')

    plt.show()


    # c_range2, g_range2= np.meshgrid(c_range2, g_range2)
    # surf = ax.plot_surface(c_range2, g_range2, c_scores, rstride=1, cstride=1, cmap=cm.coolwarm,
    #                    linewidth=0, antialiased=False)
    # ax.set_zlim(0, 1)
    #
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    #
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    #
    # plt.show()





def Analysis3(kk,cc,gg):
    data_1 = Data_Set()
    X, y,X2,y2 = data_1.Build_Data_Set()


    clf = svm.SVC(kernel=kk, C= cc, gamma=gg)
    clf.fit(X[:],y[:])
    correct_count = 0

    for x in range(1, len(X2[:])):
        if clf.predict(X2[x])[0] == y2[x]:
            correct_count += 1

    print("Accuracy:", (correct_count/len(y2)) * 100.00)

    print(correct_count)
    print(len(y2))
    # print(y2.count(0))
    # print("Accuracy:", (correct_count/y2.count(0)) * 100.00)


class test():

    def __init__(self):

        list_k = ['rbf', 'poly','sigmoid']
        # list_c = np.arange(10,500,40)
        # list_c = np.arange(1,300,4)
        # list_g = np.arange(0.001,0.02,0.002)

        for i in list_k:
                    print(i,':',Analysis3(i,1,0.001))


# def test2():
#     data_1 = Data_Set()
#     X, y,X2,y2 = data_1.Build_Data_Set()
#
#
#     clf = svm.SVC(kernel=kk, C= cc, gamma=gg)
#     clf.fit(X[:],y[:])
#     correct_count = 0
#
#     for x in range(1, len(X2[:])):
#         if clf.predict(X2[x])[0] == y2[x]:
#             correct_count += 1
#
#     print("Accuracy:", (correct_count/len(y2)) * 100.00)
#
#     print(correct_count)
#     print(len(y2))
#
Analysis2_5()
# Analysis3('rbf',10,1)