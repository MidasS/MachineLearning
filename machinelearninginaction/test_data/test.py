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

    # FEATURES = ['width','avg_T','std_T','rel_T[-6]','rel_T[-5]','rel_T[-4]','rel_T[-3]','rel_T[-2]','rel_T[-1]','rel_T[0]','rel_T[1]','rel_T[2]','rel_T[3]','rel_T[4]','rel_T[5]','rel_T[6]']
    # FEATURES = ['width','avg_T','std_T','after_T[-6]','after_T[-5]','after_T[-4]','after_T[-3]','after_T[-2]','after_T[-1]','after_T[0]','after_T[1]','after_T[2]','after_T[3]','after_T[4]','after_T[5]','after_T[6]', 'before_T[-6]', 'before_T[-5]', 'before_T[-4]', 'before_T[-3]', 'before_T[-2]', 'before_T[-1]', 'before_T[0]', 'before_T[1]', 'before_T[2]', 'before_T[3]', 'before_T[4]', 'before_T[5]', 'before_T[6]']
    FEATURES = ['width','avg_T','std_T','rel_T[-6]','rel_T[-5]','rel_T[-4]','rel_T[-3]','rel_T[-2]','rel_T[-1]','rel_T[0]','rel_T[1]','rel_T[2]','rel_T[3]','rel_T[4]','rel_T[5]','rel_T[6]','after_T[-6]','after_T[-5]','after_T[-4]','after_T[-3]','after_T[-2]','after_T[-1]','after_T[0]','after_T[1]','after_T[2]','after_T[3]','after_T[4]','after_T[5]','after_T[6]', 'before_T[-6]', 'before_T[-5]', 'before_T[-4]', 'before_T[-3]', 'before_T[-2]', 'before_T[-1]', 'before_T[0]', 'before_T[1]', 'before_T[2]', 'before_T[3]', 'before_T[4]', 'before_T[5]', 'before_T[6]']
    # FEATURES = ['width','avg_T','std_T','rel_T[-3]','rel_T[-2]','rel_T[-1]','rel_T[0]','rel_T[1]','rel_T[2]','rel_T[3]','after_T[-3]','after_T[-2]','after_T[-1]','after_T[0]','after_T[1]','after_T[2]','after_T[3]']
    # FEATURES = ['width','avg_T','std_T','rel_T[-3]','rel_T[-2]','rel_T[-1]','rel_T[0]','rel_T[1]','rel_T[2]','rel_T[3]']
    # FEATURES = ['width','avg_T','std_T','after_T[-3]','after_T[-2]','after_T[-1]','after_T[0]','after_T[1]','after_T[2]','after_T[3]']






    def Build_Data_Set(self):
        data_df = pd.DataFrame.from_csv("final_data.csv")
        # data_df = data_df.reindex(np.random.permutation(data_df.index))

        self.X = np.array(data_df[self.FEATURES].values)
        # self.X = preprocessing.scale(self.X)
        self.y = (data_df["result_2"].values.tolist())

        data_df2 = pd.DataFrame.from_csv("new_test3.csv")
        data_df2 = data_df2.reindex(np.random.permutation(data_df2.index))

        self.X2 = np.array(data_df2[self.FEATURES].values)
        self.X2 = preprocessing.scale(self.X2)
        self.y2 = (data_df2["result_2"].values.tolist())


        return self.X,self.y









    def Analysis2_6(self):

        data_1 = Data_Set()
        X, y = data_1.Build_Data_Set()


        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        c_range = np.arange(1,100,0.5)
        g_range = np.arange(0.001,0.01,0.002)

        c_scores =[]

        for c in c_range:
            for g in g_range:
                clf = svm.SVC(kernel="rbf", C= c, gamma= g)
                scores = cross_val_score(clf,X,y, cv=5, scoring='accuracy')
                c_scores.append(scores.mean())

        ax.scatter(c_range, g_range, c_scores, c='r', marker='o')

        ax.set_xlabel('Value of C for SVM')
        ax.set_ylabel('Value of gamma for SVM')
        ax.set_zlabel('Accuracy')

        plt.show()

d1 = Data_Set()
print(d1.Build_Data_Set())