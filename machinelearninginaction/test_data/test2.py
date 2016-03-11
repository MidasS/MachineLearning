import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, preprocessing
import pandas as pd
from matplotlib import style
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
style.use("ggplot")



feature1 = ['a','b']
feature2 = ['c','d']

featurelist = [feature1, feature2]

for i in featurelist:
    print(featurelist[0])