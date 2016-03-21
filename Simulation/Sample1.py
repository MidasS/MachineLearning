import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, preprocessing
import pandas as pd
from matplotlib import style


data_df = pd.DataFrame.from_csv("TimeTablef.csv")

print(data_df)