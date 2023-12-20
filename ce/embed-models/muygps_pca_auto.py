
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from MuyGPyS.examples.classify import do_classify
from MuyGPyS.gp.deformation import F2, Isotropy
from MuyGPyS.gp.hyperparameter import Parameter, Parameter as ScalarParam
from MuyGPyS.gp.kernels import RBF, Matern
from MuyGPyS.gp.noise import HomoscedasticNoise
from MuyGPyS.optimize import Bayes_optimize
from MuyGPyS.optimize.loss import LossFn, cross_entropy_fn

from glob import glob
# path = '../../data/data-norm/pca-embed-auto/'

def mymuygps(path):
    norm_name = []
    my_accuracy = []
    # read normalized data csv file names from the data directory
    norm_data_names = glob(path + '*.csv')
    # get rid of "../data/data-norm/"
    norm_data_names = [name.split('/')[-1] for name in norm_data_names]
    # norm_data_names[:10]

    # sort the names by their numbers
    norm_data_names.sort(key=lambda x: x.split('_')[1])

    nn_kwargs_exact = {"nn_method": "exact", "algorithm": "ball_tree"}

    nn_kwargs_hnsw = {"nn_method": "hnsw"}

    k_kwargs_rbf ={
                "kernel": RBF(
                    deformation=Isotropy(
                        metric=F2,
                    length_scale=Parameter(1.0, (1e-2, 1e2)),
                    ),
                ),
                "noise": HomoscedasticNoise(1e-5),
                }
    k_kwargs_mattern= { "kernel": Matern(
                smoothness=ScalarParam(0.5),
                deformation=Isotropy(
                    metric=F2,
                    length_scale=Parameter(1.0, (1e-2, 1e2)),
                ),
            ),
            "noise": HomoscedasticNoise(1e-5),
            }
    for name in tqdm(norm_data_names):
        path1 = path + name
        data = pd.read_csv(path1,na_values='-')
        data.fillna(0,inplace=True)
        data_label = path1.split('/')[-1]
        truth_labels = data.iloc[:, 0].values
        image_data = data.iloc[:, 1:].values

        X_train, X_test, y_train, y_test = train_test_split(image_data, truth_labels, test_size=0.2, random_state=42)

        print("=============== ", data_label, " ===============")
        print('Training data:', len(y_train[y_train==0]), 'single stars and', len(y_train[y_train==1]), 'blended stars')
        print('Testing data:', len(y_test[y_test==0]), 'single stars and', len(y_test[y_test==1]), 'blended stars')

        onehot_train, onehot_test = generate_onehot_value(y_train), generate_onehot_value(y_test)

        train = {'input': X_train, 'output': onehot_train, 'lookup': y_train}
        test = {'input': X_test, 'output': onehot_test, 'lookup': y_test}

        print("Running Classifier on", data_label)
        #Switch verbose to True for more output


        muygps, nbrs_lookup, surrogate_predictions = do_classify(
                                    test_features=np.array(test['input']), 
                                    train_features=np.array(train['input']), 
                                    train_labels=np.array(train['output']), 
                                    nn_count=30,
                                    batch_count=200,
                                    loss_fn=cross_entropy_fn,
                                    opt_fn=Bayes_optimize,
                                    k_kwargs=k_kwargs_mattern,
                                    nn_kwargs=nn_kwargs_hnsw,
                                    verbose=False)
        predicted_labels = np.argmax(surrogate_predictions, axis=1)
        accur = np.around((np.sum(predicted_labels == np.argmax(test["output"], axis=1))/len(predicted_labels))*100, 3),
        norm_name.append("".join(data_label.split('.')[:-1]))
        my_accuracy.append(accur[0])
        print("Total accuracy for", data_label, ":", accur, '%')
    return norm_name, my_accuracy



def generate_onehot_value(values):
    onehot = []
    for val in values:
        if val == 0:
            onehot.append([1., -1.])
        elif val == 1:
            onehot.append([-1., 1.])
    return onehot



if __name__ == "__main__":
    # run muygps over  pca embedded data across all normalization techniques
    from tqdm import tqdm
    from time import perf_counter

    t_start = perf_counter()
    n = 50
    results = {}
    for i in tqdm(range(2, n+1)):
        path = f'../../data/data-norm/pca-embed-max-only/pca-ncomp-{i}/'
        norm_name, my_accuracy = mymuygps(path)
        results[f'{i}-cps'] = my_accuracy
    result = pd.DataFrame(results, index=norm_name)
    result.to_csv('muygps_pca_embeded-results.csv', index=True)
    t_end = perf_counter()
    print(f'Finished in {(t_end-t_start)/60} minutes')
