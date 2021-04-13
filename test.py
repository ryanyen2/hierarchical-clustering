import pandas as pd
import numpy as np
from sklearn import preprocessing
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import LocalOutlierFactor


vertebral_data = pd.read_csv('vertebral_column_data/column_3C.dat', header=None, sep=" ").values
print(max(vertebral_data[:, -2]))
# for i, v in enumerate(vertebral_data):