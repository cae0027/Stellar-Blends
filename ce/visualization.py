
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from time import perf_counter


# Read csv file with the labels and image data
data_path = '../data/data-norm/max-only/raw_image_data.csv'
def embedding(data_path=data_path, pltt=False):
    now = perf_counter()
    data = pd.read_csv(data_path)
    # count how many rows with NAN values
    print("Total number of rows with NAN values: ", np.count_nonzero(data.isnull()))
    # drop rows with NAN values
    data.dropna(inplace=True)
    # get label
    label = data.iloc[:, 0].astype(int)
    label = np.array([int(x) for x in label])
    image_data = data.iloc[:,1:]

    # perform pca on data
    pca = PCA(n_components=2)
    pca_fit = pca.fit_transform(image_data)

    # perform tSNE on data
    tsne = TSNE(n_components = 2, random_state=0)
    tsne_fit = tsne.fit_transform(image_data)

    # perform unsupervised umap on data
    umap_fit = umap.UMAP().fit_transform(image_data)

    # split data for supervised umap
    X_train, X_test, y_train, y_test = train_test_split(image_data, label, test_size=0.3, random_state=42)
    # umap train
    umap_sup = umap.UMAP().fit(X_train, y=y_train)
    sup_umap_train = umap_sup.transform(X_train)
    #umap test
    sup_umap_test = umap_sup.transform(X_test)



    # plot and save results
    fig, axes = plt.subplots(nrows=1, ncols=5, sharex=False, sharey=False, constrained_layout=True, figsize=(15, 3))
    # , figsize=(12, 8)

    columns = ['PCA', 't-SNE', 'UMAP', 'Supervised UMAP-Train', 'Supervised UMAP-Test']
    results = [pca_fit, tsne_fit, umap_fit, sup_umap_train, sup_umap_test]

    # labels for umap
    labels_dict = {'UMAP': label, 'Supervised UMAP-Train': y_train, 'Supervised UMAP-Test': y_test}


    for i, name in enumerate(results):
        if i < 2:
            sns.scatterplot(x = name[:,0], y = name[:,1], hue = label, legend = 'full', ax=axes[i])
            axes[i].set_title(columns[i])
        else:
            label_u = labels_dict[columns[i]]
            axes[i].scatter(name[:, 0], name[:, 1], c=label_u, cmap='Spectral', s=5)
            plt.gca().set_aspect('equal', 'datalim')
            axes[i].set_title(columns[i])
    data_name = ''.join(data_path.split('/')[-1].split('.')[:2])
    # plt.savefig('./embed-results/max-only/'+data_name, dpi=500)
    plt.savefig('./embed-results/max-pixel-all/'+data_name, dpi=500)

    end = perf_counter()
    print(f"Total time taken for embedding a single data is {(end-now)/60} minutes")
    if pltt:
        plt.show()
    else:
        return


if __name__ == '__main__':
    from glob import glob
    now = perf_counter()

    # read normalized data csv file names from the data directory
    # norm_data_names = glob('../data/data-norm/max-only/*.csv')
    norm_data_names = glob('../data/data-norm/max-pixel-all/*.csv')
    # sort the names by their numbers
    norm_data_names.sort(key=lambda x: x.split('_')[1])

    for i,name in enumerate(norm_data_names):
        print(f"Running Iteration {(i+1)}/{len(norm_data_names)} ================> {name}")
        embedding(data_path=name)

    # run single data file embedding
    # embedding(data_path=data_path)

    end = perf_counter()
    print(f"Total time taken for embedding all normalized data is {(end-now)/60} minutes")

