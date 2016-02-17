from scipy.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, preprocessing
import pandas as pd
from matplotlib import style
style.use("ggplot")
from machinelearninginaction.test_data.Ucim import Data_Set


d = Data_Set()
X,y,X2,y2 = d.Build_Data_Set()
print(X,y,X2,y2 )
# pearsonr()