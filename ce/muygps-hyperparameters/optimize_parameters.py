#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from MuyGPyS.examples.classify import do_classify
from MuyGPyS.gp.deformation import F2, Isotropy, l2
from MuyGPyS.gp.hyperparameter import Parameter, Parameter as ScalarParam
from MuyGPyS.gp.kernels import RBF, Matern
from MuyGPyS.gp.noise import HomoscedasticNoise
from MuyGPyS.optimize import Bayes_optimize, OptimizeFn, L_BFGS_B_optimize
from MuyGPyS.optimize.loss import LossFn, cross_entropy_fn
from time import perf_counter
from MuyGPyS.optimize.loss import mse_fn, lool_fn, pseudo_huber_fn, looph_fn
import umap
import pickle
from glob import glob


# data_path = ['norm_11.csv','norm_1.csv', 'norm_21.csv', 'raw_image_data.csv']
# data_path = [ 'norm_21.csv', 'raw_image_data.csv','nthroot_mm0.3103.csv','nthroot_mm0.3793.csv', 'nthroot_mm0.4138.csv']
data_path = glob('../../data/data-norm/max-only-best/*.csv')
norm_data_names = [name.split('/')[-1] for name in data_path] 

def choose_components():
    """
    Fixed number of components for each data set, initially defined based on the fact that the number of components have been optimized for these data sets for UMAP embedding
    """
    if data_label == 'norm_21':
            return 31
    elif data_label == 'raw_image_data':
        return 8
    elif data_label == 'nthroot_mm03103':
        return 4
    elif data_label == 'nthroot_mm03793':
        return 37
    elif data_label == 'nthroot_mm04138':
        return 8
    return 2

def load_data(path,embedding, rand_ncomponents=False):
    path1 = '../../data/data-norm/max-only-best/' + path
    data = pd.read_csv(path1,na_values='-')
    data.fillna(0,inplace=True)
    data_label = ''.join(path.split('.')[:-1])
    truth_labels = data.iloc[:, 0].values
    image_data = data.iloc[:, 1:].values
    # split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(image_data, truth_labels, test_size=0.2, random_state=42)
    embed_type = 'none' # indicates no embedding
    n = 0   # indicates no embedding
    # don't want to embed every time, so only do it 50% of the time
    choice = random.choice([embedding, 1])
    if choice==embedding:
        # embed data using umap
        # choose different number of components for different data
        if rand_ncomponents:
            n = random.choice(np.arange(2, 100))
        else:
            n = choose_components()
        # include PCA embedding, so make a choice between PCA and UMAP
        if random.choice([True, False]):
            pca = PCA(n_components=n)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)
            embed_type = 'pca'
        else:
            umapped = umap.UMAP(n_components=n).fit(X_train, y_train)
            X_train = umapped.transform(X_train)
            X_test = umapped.transform(X_test)
            embed_type = 'umap'
    return X_train, X_test, y_train, y_test, data_label, choice, n, embed_type


def generate_onehot_value(values):
    onehot = []
    for val in values:
        if val == 0:
            onehot.append([1., -1.])
        elif val == 1:
            onehot.append([-1., 1.])
    return onehot


nn_kwargs_exact = {"nn_method": "exact", "algorithm": "ball_tree"}

nn_kwargs_hnsw = {"nn_method": "hnsw"}

# choose kernel and feed the corresponding parameters
def choose_kernel(kernel, metric, length_scale, noise, smoothness):
    assert kernel in ['rbf', 'mattern'], "kernel must be either 'rbf' or 'mattern'"
    if kernel == 'rbf':
        return {"kernel": RBF(
            deformation=Isotropy(
                metric=metric,
                length_scale=Parameter(length_scale, (1e-2, 1e2)),
            ),
        ),
        "noise":HomoscedasticNoise(noise),
        }
    else:
        return {"kernel": Matern(
            smoothness=ScalarParam(smoothness),
            deformation=Isotropy(
                metric=metric,
                length_scale=Parameter(length_scale, (1e-2, 1e2)),
            ),
        ),
        "noise":HomoscedasticNoise(noise),
        }

loss_funcs = [mse_fn, lool_fn, pseudo_huber_fn, looph_fn, cross_entropy_fn]
# string names for loss functions
loss_names = ['mse_fn', 'lool_fn', 'pseudo_huber_fn', 'looph_fn', 'cross_entropy_fn']
optimizers = [Bayes_optimize, L_BFGS_B_optimize]
# string names for optimizers
opt_names = ['Bayes_optimize', 'L_BFGS_B_optimize']
kernels = ['rbf', 'mattern']
nn_methods = [nn_kwargs_exact, nn_kwargs_hnsw]
metrics = [l2, F2]
# string names for metrics
metric_names = ['l2', 'F2']
# explore two different ranges for length scales
length_scales = np.linspace(1e-2, 1e2, 50) if random.choice([True, False]) \
      else np.linspace(1e-2, 1, 50)
smoothnesses = np.linspace(1e-2, 1, 50) # for Matern kernel only
# explore two different ranges for homoscedastic noise
homosc_noise = np.linspace(1e-7, 1e0, 50) if random.choice([True, False]) \
        else np.linspace(1e0, 1e2, 50)
embedding = 'embed'   # umap embedding since it's independent of the components
batch_counts = [ 5,10,20,40,80,160,320,640]
nn_counts = [5, 10, 15, 20,25,30,35,40 ]


# number of random runs for hyperparameter optimization
m = 10
accuracies = {'Length Scale': [], 'Batch Count': [], 'NN Count': [], 'Accuracy': [], 'Loss': [], 'Time': [], 'Kernel': [], 'NN Method': [], 'Metric': [], 'Smoothness': [], 'Homoscedastic Noise': [], 'Embedding': [], 'Numb-comps':[], 'Data': [], 'Optimizer': []}

for _ in tqdm(range(m)):
    for path in norm_data_names:
            # load data
            X_train, X_test, y_train, y_test, data_label, choice, n, embed_type = load_data(path, embedding, rand_ncomponents=True)
            # get random hyperparameters
            loss_idx = random.choice(range(len(loss_funcs)))
            loss = loss_funcs[loss_idx]
            optimizer_idx = random.choice(range(len(optimizers)))
            optimizer = optimizers[optimizer_idx]
            nn_method = random.choice(nn_methods)
            kernel = random.choice(kernels)
            metric_idx = random.choice(range(len(metrics)))
            metric = metrics[metric_idx]
            ls = random.choice(length_scales)
            smoothness = random.choice(smoothnesses)
            hn = random.choice(homosc_noise)
            batch = random.choice(batch_counts)
            nn = random.choice(nn_counts)
            # update k_kwargs
            k_kwargs = choose_kernel(kernel, metric, ls, hn, smoothness)

            print("=============== ", data_label, " ===============")
            print('Training data:', len(y_train[y_train==0]), 'single stars and', len(y_train[y_train==1]), 'blended stars')
            print('Testing data:', len(y_test[y_test==0]), 'single stars and', len(y_test[y_test==1]), 'blended stars')

            onehot_train, onehot_test = generate_onehot_value(y_train), generate_onehot_value(y_test)

            train = {'input': X_train, 'output': onehot_train, 'lookup': y_train}
            test = {'input': X_test, 'output': onehot_test, 'lookup': y_test}

            print("Running Classifier on", data_label)

            # run classifier
            start = perf_counter()
            muygps, nbrs_lookup, surrogate_predictions = do_classify(
                                            test_features=np.array(test['input']), 
                                            train_features=np.array(train['input']), 
                                            train_labels=np.array(train['output']), 
                                            nn_count=nn,
                                            batch_count=batch,
                                            loss_fn=loss,
                                            opt_fn=optimizer,
                                            k_kwargs=k_kwargs,
                                            nn_kwargs=nn_method,
                                            verbose=False
                                            )
            end = perf_counter()
            predicted_labels = np.argmax(surrogate_predictions, axis=1)
            accur = np.around((np.sum(predicted_labels == np.argmax(test["output"], axis=1))/len(predicted_labels))*100, 3)

            # update accuracies dictionary
            accuracies['Time'].append(end-start)
            accuracies['Length Scale'].append(ls)
            accuracies['Batch Count'].append(batch)
            accuracies['NN Count'].append(nn)
            accuracies['Loss'].append(loss_names[loss_idx])
            accuracies['Kernel'].append(kernel)
            accuracies['NN Method'].append(nn_method['nn_method'])
            accuracies['Metric'].append(metric_names[metric_idx])
            accuracies['Smoothness'].append(smoothness)
            accuracies['Homoscedastic Noise'].append(hn)
            accuracies['Embedding'].append(embed_type)
            accuracies['Numb-comps'].append(n)
            accuracies['Data'].append(data_label)
            accuracies['Optimizer'].append(opt_names[optimizer_idx])
            accuracies['Accuracy'].append(accur)

            print("Total accuracy for", data_label, ":", accur, '%')
            print("=====================================================")
        # save accuracies to csv
    df_new = pd.DataFrame(accuracies)
    # load the previous results and merge with the new results
    # be sure to check if the previous results file exists first
    try:
        with open('accuracies-over-all-hyperparameters-pca-umap.pkl', 'rb') as f:
            df_old = pickle.load(f)
        df = pd.concat([df_old, df_new], ignore_index=True)
        # pickle dump the results while model is running
        with open('accuracies-over-all-hyperparameters-pca-umap.pkl', 'wb') as f:
            pickle.dump(df, f)
    except:
        with open('accuracies-over-all-hyperparameters-pca-umap.pkl', 'wb') as f:
            pickle.dump(df_new, f)




