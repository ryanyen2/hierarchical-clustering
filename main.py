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


def getClusterInfo(n_clusters, cluster_info):
    k_clusters = fcluster(cluster_info, n_clusters, criterion='maxclust')

    # build a table to show the number of count
    clusterCount = []
    for i in range(0, n_clusters):
        clusterCount.append(0)
    for i in range(0, len(k_clusters)):
        clusterCount[k_clusters[i] - 1] += 1
    for i in range(0, n_clusters):
        print(str(i + 1) + ": " + str(clusterCount[i]))
    return k_clusters


def remove_outlier(data, label):
    lof = LocalOutlierFactor()
    yhat = lof.fit_predict(data)

    mask = yhat != -1

    removed_outlier = []
    for i, v in enumerate(mask):
        if not v:
            removed_outlier.append(i)
    print("Removed Outlier index: ", removed_outlier)

    data, label = data[mask, :], label[mask]
    print("Left (data, label): ", data.shape, label.shape)
    return data, label


def min_max_normalized(d):
    min_max_scalar = preprocessing.MinMaxScaler(feature_range=(0, 1))
    x_scaled = min_max_scalar.fit_transform(d)

    return x_scaled


def plot_single_link(xs):
    single_link = linkage(xs, 'single')
    fig = plt.figure(figsize=(100, 80))
    dn = dendrogram(single_link)

    plt.xticks(fontsize=18)
    plt.savefig('output/single_link.png')

    print(getClusterInfo(3, single_link))
    # for i in range(0, len(single_link)):
    #     print(single_link[i])


def plot_complete_link(xs):
    complete_link = linkage(xs, 'complete')
    fig = plt.figure(figsize=(100, 80))

    plt.xticks(fontsize=20)
    dn = dendrogram(complete_link)
    plt.savefig('output/complete_link.png')

    print(getClusterInfo(3, complete_link))


def plot_groups_average(xs):
    average_link = linkage(xs, 'average')
    fig = plt.figure(figsize=(100, 80))
    plt.xticks(fontsize=20)
    dn = dendrogram(average_link)
    plt.savefig('output/average_link.png')


def scatter_plot_data(d, l, d1, d2):
    le = preprocessing.LabelEncoder()
    label = le.fit_transform(l)

    feature_names = ["pelvic incidence", "pelvic tilt", "lumbar lordosis angle", "sacral slope", "pelvic radius",
                     "grade of spondylolisthesis"]
    class_names = ["Disk Hernia (DH)", "Spondylolisthesis (SL)", "Normal (NO)"]
    plt.scatter(d[:, d1].astype(float), d[:, d2].astype(float), c=label, alpha=0.4, cmap='viridis', label=class_names)

    plt.xlabel(feature_names[d1])
    plt.ylabel(feature_names[d2])
    # 0 = DH/ 1 = NO/ 2 = SL
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    # feature_names =["pelvic incidence","pelvic tilt","lumbar lordosis angle", "sacral slope", "pelvic radius", "grade of spondylolisthesis"]
    # class_names=["Disk Hernia (DH)", "Spondylolisthesis (SL)", "Normal (NO)"]
    vertebral_data = pd.read_csv('vertebral_column_data/column_3C.dat', header=None, sep=" ").values

    data, label = vertebral_data[:, :-1], vertebral_data[:, -1]
    print("Original (data, label): ", data.shape, label.shape)
    # scatter_plot_data(data, label, 3, 5)

    # remove outlier
    data, label = remove_outlier(data, label)
    # scatter_plot_data(data, label, 3, 5)

    # normalize
    X_scaled = min_max_normalized(data)
    scatter_plot_data(X_scaled, label, 3, 5)

    # single link
    # plot_single_link(X_scaled)
    # plot_complete_link(X_scaled)
    # plot_groups_average(X_scaled)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
