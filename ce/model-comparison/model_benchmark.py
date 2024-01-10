
from glob import glob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
# from sklearn.inspection import DecisionBoundaryDisplay
import umap

# path = '../data/data-norm/max-only/'
path = '../data/data-norm/max-pixel-all/'
data_names = glob(path + '*.csv')
bench_mark_results = {name.split('/')[-1]:[] for name in data_names  }
for data_name in bench_mark_results:
    data = pd.read_csv(path + data_name)
    # Get the label data
    label = data.iloc[:,0]
    # Use .iloc to locate the image data within the DataFrame
    image_data = data.iloc[:, 1:]

    names = [
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    "Gaussian Process",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "QDA",
    ]

    # KNeighborsClassifier 0.6862043661713447
    # SVC 0.5730141258484682
    # SVC 0.7052834342322509
    # GaussianProcessClassifier
    # DecisionTreeClassifier 0.6511649238671803
    # RandomForestClassifier 0.6606127316088791
    # MLPClassifier 0.6231884057971014
    # AdaBoostClassifier 0.652357365620987
    # GaussianNB 0.5719134103834159
    # QuadraticDiscriminantAnalysis 0.5969546872133553

    classifiers = [
        # KNeighborsClassifier(3),
        # KNeighborsClassifier(5),
        # KNeighborsClassifier(10),
        # SVC(kernel="linear", C=0.025),
        # SVC(gamma=1, C=1),
        # SVC(gamma=2, C=1),
        # SVC(gamma=4, C=1),
        # GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        # RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        # MLPClassifier(alpha=1, max_iter=1000),
        # AdaBoostClassifier(),
        # GaussianNB(),
        # QuadraticDiscriminantAnalysis(),
    ]

    X, y = image_data, label
    # X, y = data2.iloc[:,2:], label2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # reducer = umap.UMAP(min_dist=0,n_neighbors=100,n_components=20).fit(X_train, y=y_train)
    # embedding_train = reducer.transform(X_train)
    # embedding_test = reducer.transform(X_test)

    # iterate over classifiers
    # for name, clf in zip(names, classifiers):
    #     # ax = plt.subplot(1, len(classifiers) + 1, i)
    #     clf.fit(embedding_train, y_train)
    #     score = clf.score(embedding_test, y_test)
    #     print(clf.__class__.__name__, score)


    for name, clf in zip(names, classifiers):
        # ax = plt.subplot(1, len(classifiers) + 1, i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print(clf.__class__.__name__, score)
    bench_mark_results[data_name].append(score)
bench_mark_results = pd.DataFrame(bench_mark_results)
bench_mark_results.to_csv('bench_mark_results_maxall.csv', index=False)


