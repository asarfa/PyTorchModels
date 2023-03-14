from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import random_split
from torchsummary import summary
import time
import numpy as np
import random
import pandas as pd

from src.models import CNN
from src.trainer import Trainer
import torchvision
import torchvision.transforms as transforms
import requests
import os


if __name__ == '__main__':
    """
    Loading dataset

    import torchvision
    import torchvision.transforms as transforms
    import requests
    import os
    """

    if not os.path.exists('USPS/usps.bz2'):
        print("Data not found downloading it!")
        url = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.bz2'
        r = requests.get(url, allow_redirects=True)
        if not os.path.isdir('USPS/'):
            os.mkdir('USPS/')
        open('USPS/usps.bz2', 'wb').write(r.content)
        print("Download finished!")

    print("Loading data ...")
    dataset = torchvision.datasets.USPS(root='USPS/',
                                        train=True,
                                        transform=transforms.ToTensor(),
                                        download=False)

    train_set, val_set = random_split(dataset, [6000, 1291])

    """
    defining space of hyper parameters
    """
    hidden_dim = [5, 10, 20]
    n_hidden = [1, 2, 3, 5]
    activation = ["sigmoid", "relu", "tanh"]
    kernel_size = [1, 2, 3]
    batch_size = [10, 100, 300, 500, 1000]
    learning_rate = [1e-4, 5e-5, 1e-3, 1e-2]
    optimizer = ['Adam', 'SGD', 'RMS']

    """
    Defining input/output shape
    """
    input_shape = (16, 16)
    output_dim = 10

    """
    Combinations of all hyper parameters
    """
    mshgrd = np.array(
        np.meshgrid(hidden_dim, n_hidden, activation, kernel_size, batch_size, learning_rate, optimizer)).T.reshape(-1,
                                                                                                                    7)
    dict_param = dict(map(lambda i: (i, {'hidden_dim': int(mshgrd[i][0]),
                                         'n_hidden': int(mshgrd[i][1]),
                                         'activation': mshgrd[i][2],
                                         'kernel_size': int(mshgrd[i][3]),
                                         'batch_size': int(mshgrd[i][4]),
                                         'lr': float(mshgrd[i][5]),
                                         'optimizer': mshgrd[i][6]}),
                          range(len(mshgrd))))

    print(f'number of hyperparameters combination: {len(dict_param)}')

    print(f'Naive model with: {dict_param[0]}')
    model = CNN(input_shape=input_shape, output_dim=output_dim,
                params=dict_param[0], last_activation=False)
    print(model)

    """
    Defining max number of hyperoptimization trials
    """
    max_combinations = 10
    random_key = random.sample(list(dict_param), max_combinations)

    """
    Finding best hyperparameters over max number of trials/combination
    Sorted accorded maximum validation accuracy
    Saving values for each trial of hyperparameters set
    """
    results = pd.DataFrame(
        columns=['train_acc', 'val_acc', 'hidden_dim', 'n_hidden', 'activation', 'kernel_size', 'batch_size',
                 'learning_rate', 'optimizer', 'best_epoch'], index=range(len(random_key)))
    best_val_acc = -np.inf
    best_params = None
    start = time.time()
    for i, key in tqdm(enumerate(random_key), total=max_combinations):
        params = dict_param[key]
        params_architecture = dict(list(params.items())[:4])
        model = Trainer(CNN(input_shape=input_shape, output_dim=output_dim, params=params_architecture,
                            last_activation=False),
                        verbose=True, lr=params['lr'],
                        opt=params['optimizer'], batch_size=params['batch_size'],
                        epochs=50, criterion=nn.CrossEntropyLoss())
        model.fit(train_set, val_set)
        train_acc = round(model.train_acc, 4)
        val_acc = round(model.val_acc, 4)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = params
        results.iloc[i] = [train_acc, val_acc] + \
            list(params.values()) + [model.best_epoch]
    print(f'Time elapsed: {round((time.time() - start) / 60, 3)} minutes')
    results = results.sort_values('val_acc', ascending=False)
    print('Hyper parameters sorted according maximum validation accuracy: ')
    print(results)

    """
    Loading test set
    
    """
    if os.path.exists('USPS/usps.t.bz2'):
        print("Data not found!")
        print("Downloading it ...")
        url = 'http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.t.bz2'
        r = requests.get(url, allow_redirects=True)
        open('USPS/usps.t.bz2', 'wb').write(r.content)
        print("Test data downloaded")
    print("Loading data ...")
    test_set = torchvision.datasets.USPS(root='USPS/',
                                         train=False,
                                         transform=transforms.ToTensor(),
                                         download=False)
    test_set = None
    """
    Predicting on test set with best hyper params set
    """
    if best_params:
        print(f'Best hyper params: {best_params}')
        best_params_architecture = dict(list(best_params.items())[:4])
        best_model = Trainer(CNN(input_shape=input_shape, output_dim=output_dim, params=best_params_architecture,
                                 last_activation=False),
                             verbose=False, lr=best_params['lr'],
                             opt=best_params['optimizer'], batch_size=best_params['batch_size'],
                             epochs=50, criterion=nn.CrossEntropyLoss())
        summary(best_model.model, (1, input_shape[0], input_shape[1]))
        best_model.fit(train_set, val_set)
        test_pred = best_model.predict(test_set)

    else:
        print("No best parameters found over there!")
