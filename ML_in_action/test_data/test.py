import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, preprocessing
import pandas as pd
from matplotlib import style
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
style.use("ggplot")
import pylab, random


FEATURES1 = ['avg_T','std_T','rel_T[-6]','rel_T[-5]','rel_T[-4]','rel_T[-3]','rel_T[-2]','rel_T[-1]','rel_T[0]','rel_T[1]','rel_T[2]','rel_T[3]','rel_T[4]','rel_T[5]','rel_T[6]','after_T[-6]','after_T[-5]','after_T[-4]','after_T[-3]','after_T[-2]','after_T[-1]','after_T[0]','after_T[1]','after_T[2]','after_T[3]','after_T[4]','after_T[5]','after_T[6]', 'before_T[-6]', 'before_T[-5]', 'before_T[-4]', 'before_T[-3]', 'before_T[-2]', 'before_T[-1]', 'before_T[0]', 'before_T[1]', 'before_T[2]', 'before_T[3]', 'before_T[4]', 'before_T[5]', 'before_T[6]']
FEATURES2 = ['avg_T','std_T','rel_T[-3]','rel_T[-2]','rel_T[-1]','rel_T[0]','rel_T[1]','rel_T[2]','rel_T[3]','after_T[-3]','after_T[-2]','after_T[-1]','after_T[0]','after_T[1]','after_T[2]','after_T[3]']
FEATURES3 = ['avg_T','std_T','rel_T[-6]','rel_T[-5]','rel_T[-4]','rel_T[-3]','rel_T[-2]','rel_T[-1]','rel_T[0]','rel_T[1]','rel_T[2]','rel_T[3]','rel_T[4]','rel_T[5]','rel_T[6]']
FEATURES4 = ['avg_T','std_T','rel_T[-3]','rel_T[-2]','rel_T[-1]','rel_T[0]','rel_T[1]','rel_T[2]','rel_T[3]']
FEATURES5 = ['after_T[-6]','after_T[-5]','after_T[-4]','after_T[-3]','after_T[-2]','after_T[-1]','after_T[0]','after_T[1]','after_T[2]','after_T[3]','after_T[4]','after_T[5]','after_T[6]', 'before_T[-6]', 'before_T[-5]', 'before_T[-4]', 'before_T[-3]', 'before_T[-2]', 'before_T[-1]', 'before_T[0]', 'before_T[1]', 'before_T[2]', 'before_T[3]', 'before_T[4]', 'before_T[5]', 'before_T[6]']
FEATURES6 = ['after_T[-3]','after_T[-2]','after_T[-1]','after_T[0]','after_T[1]','after_T[2]','after_T[3]']

class Data_Set():

    def Build_Data_Set(self,FEATURES):
        data_df = pd.DataFrame.from_csv("final_data.csv")
        # data_df = data_df.reindex(np.random.permutation(data_df.index))

        self.X = np.array(data_df[FEATURES].values)
        # self.X = preprocessing.scale(self.X)
        self.y = (data_df["result_2"].values.tolist())

        # data_df2 = pd.DataFrame.from_csv("new_test3.csv")
        # data_df2 = data_df2.reindex(np.random.permutation(data_df2.index))
        #
        # self.X2 = np.array(data_df2[self.FEATURES].values)
        # self.X2 = preprocessing.scale(self.X2)
        # self.y2 = (data_df2["result_2"].values.tolist())
        #

        return self.X,self.y




    def Analysis2_6(self,t, DS):

        data_1 = Data_Set()
        X, y = data_1.Build_Data_Set(DS)
        print(DS)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # c_range = np.arange(1e-2, 1, 1e2)
        # g_range = np.arange(1e-1, 1, 1e1)

        k_list = ['poly']
        c_range = [0.1, 1, 10, 30, 50 , 70, 100]
        # g_range = [0.1, 0.01, 0.001, 0.003, 0.006, 0.008, 0.0001]
        g_range = [0.005, 0.003, 0.001, 0.0005, 0.0001, 0.00001]

        parameters = dict(kernel=k_list ,C= c_range, gamma=g_range)

        grid = GridSearchCV(svm.SVC(),parameters, cv=5)
        grid.fit(X,y)
        print("The best parameters are %s with a score of %0.4f"% (grid.best_params_, grid.best_score_))

        c_scores =[]

        for c in c_range:
            for g in g_range:
                clf = svm.SVC(kernel="poly", C= c, gamma= g)
                scores = cross_val_score(clf,X,y, cv=5, scoring='accuracy')
                c_scores.append((c, g, scores.mean()))
        #
        #         clf.fit(X[:],y[:])
        #         correct_count = 0
        #         type1 = 0
        #         type2 = 0
        #
        #
        #         for x in range(1, len(X[:])):
        #             if clf.predict(X[x])[0] == y[x]:
        #                 correct_count += 1
        #             elif clf.predict(X[x])[0] == 0 and y[x] == 1:
        #                 type1 += 1
        #             elif clf.predict(X[x])[0] == 1 and y[x] == 0:
        #                 type2 += 1
        #
        #
        #         print("C:",c, "gamma:",g,"Accuracy:", (correct_count/len(y)) * 100.00)
        #         print(correct_count)
        #         print(len(y))
        #         print(type1)
        #         print(type2)

        c2 =[]
        g2 = []
        acc2 =[]

        for (k, (self.C, self.g, self.acc)) in enumerate(c_scores):
        # for C, g, acc in c_scores:
            c2.append(self.C)
            g2.append(self.g)
            acc2.append(self.acc)




        # ax.scatter(c2, g2, acc2, c='r', marker='o')
        # ax.set_xlabel('Value of C for SVM')
        # ax.set_ylabel('Value of gamma for SVM')
        # ax.set_zlabel('Accuracy')
        # ax.set_axis_bgcolor('white')
        # plt.title('rbf Type%s accuracy according to C & gamma' %(t+1))
        # plt.draw()

c1 = Data_Set()

FEATURESLIST = [FEATURES1, FEATURES2, FEATURES3, FEATURES4, FEATURES5, FEATURES6, ]

for t, f in enumerate(FEATURESLIST):
    c1.Analysis2_6(t,f)
    print(t)

