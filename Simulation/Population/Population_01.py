import numpy as np
import pandas as pd
from matplotlib import style

style.use("ggplot")

FEATURES = ['RND_ratio','GDP_per_Asset','Kospi','Patent_per_Million','LP_per_Hour','PL_ratio','LB_ratio','IAD_ratio','Emp_ratio','GNP_per_Man','GDP_per_Cab','GNI_per_Trade','GDP_per_Inv','1_IND','2_IND','3_IND','PBR_per_man','IS_ratio','CPI','CE_per_Man','WAGE_Gap','GINI','BR_per_1000','DR_per_1000','Migration']

class Data_Set():

    def Build_Data_Set(self,FEATURES):
        self.df = pd.DataFrame.from_csv("Data_set_03.csv")
        # data_df = data_df.reindex(np.random.permutation(data_df.index))
        print(self.df)

        self.X = np.array(self.df[FEATURES].values)
        # self.X = preprocessing.scale(self.X)
        self.y = (self.df["BR_per_1000"].values.tolist())


        return self.X,self.y

c1 = Data_Set()
# print(c1.Build_Data_Set(FEATURES))
c1.Build_Data_Set(FEATURES)
print(c1.df.Kospi)

