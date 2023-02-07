import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torchsummary import summary
import time
import numpy as np
import random
import pandas as pd
import math

from src.models import Regressor
from src.trainer import Trainer


def min_max_scale(X):
    X = np.array(X)
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    normalized_X = (X - X_min) / (X_max - X_min)
    return normalized_X, X_min, X_max


def undo_scale(normalized_array, range):
    X_min, X_max = range
    return normalized_array * (X_max - X_min) + X_min


def form_norm_set(X, y):
    normalized_X, _, _ = min_max_scale(X)
    normalized_y, y_min, y_max = min_max_scale(y)
    norm_set = TensorDataset(torch.from_numpy(normalized_X).float(),
                             torch.from_numpy(normalized_y).float())
    return norm_set, (y_min, y_max)


class GaussianLLLoss(nn.Module):
    """
    Gaussian negative log likelihood loss with outputs representing μ(xi,..,n) and log(σ(xi,..,n)^2)
    """

    def __init__(self):
        super(GaussianLLLoss, self).__init__()

    def forward(self, outputs, targets):
        means, logvars = outputs[:, 0], outputs[:, 1]
        loss = torch.sum(math.log(2 * math.pi) + logvars +
                         (targets - means) ** 2 / torch.exp(logvars))
        return torch.mean(1 / 2 * loss)


if __name__ == '__main__':
    """
    Loading dataset
    """
    X_train, X_val, X_test, y_train, y_val, y_test = None, None, None, None, None, None
    norm_train_set, y_train_range = form_norm_set(X_train, y_train)
    norm_val_set, y_val_range = form_norm_set(X_val, y_val)
    norm_test_set, y_test_range = form_norm_set(X_test, y_test)

    """
    defining space of hyper parameters
    """
    hidden_dim = [32, 64, 128, 256]
    n_hidden = np.arange(1, 5)
    activation = ["sigmoid", "relu", "tanh"]
    batch_size = [10, 50, 100, 200, 400]
    learning_rate = [1e-3, 1e-2, 1e-1, 1, 1e1]
    optimizer = ['Adam', 'SGD', 'RMS']

    """
    Defining input/output shape
    """
    input_dim = 4
    output_dim = 1

    """
    Combinations of all hyper parameters
    """
    mshgrd = np.array(np.meshgrid(hidden_dim, n_hidden, activation, batch_size, learning_rate, optimizer)).T.reshape(-1,
                                                                                                                     6)
    dict_param = dict(map(lambda i: (i, {'hidden_dim': int(mshgrd[i][0]),
                                         'n_hidden': int(mshgrd[i][1]),
                                         'activation': mshgrd[i][2],
                                         'batch_size': int(mshgrd[i][3]),
                                         'lr': float(mshgrd[i][4]),
                                         'optimizer': mshgrd[i][5]}),
                          range(len(mshgrd))))

    print(f'number of hyperparameters combination: {len(dict_param)}')

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
        columns=['train_rmse', 'val_rmse', 'hidden_dim', 'n_hidden', 'activation', 'batch_size', 'learning_rate',
                 'optimizer', 'best_epoch'], index=range(len(random_key)))
    best_val_rmse = np.inf
    best_params = None
    start = time.time()
    for i, key in enumerate(random_key):
        params = dict_param[key]
        params_architecture = dict(list(params.items())[:3])
        model = Trainer(Regressor(input_dim=input_dim, output_dim=output_dim, params=params_architecture),
                        verbose=False, lr=params['lr'],
                        opt=params['optimizer'], batch_size=params['batch_size'],
                        epochs=50)
        model.fit(norm_train_set, norm_val_set, y_train_range, y_val_range)
        try:
            train_rmse = round(model.train_rmse)
            val_rmse = round(model.val_rmse)
        except:
            train_rmse = val_rmse = np.inf
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_params = params
        results.iloc[i] = [train_rmse, val_rmse] + \
            list(params.values()) + [model.best_epoch]
    print(f'Time elapsed: {round((time.time() - start) / 60, 3)} minutes')
    results = results.sort_values('val_rmse')[:20]
    print('Top 20 hyper parameters leading to minimum RMSE validation: ')
    print(results)

    """
    Loading test set
    """
    print(f'Best hyper params: {best_params}')
    best_params_architecture = dict(list(best_params.items())[:3])
    best_model = Reg(Model(input_dim=input_dim, output_dim=output_dim, params=params_architecture),
                     verbose=False, lr=best_params['lr'],
                     opt=best_params['optimizer'], batch_size=best_params['batch_size'],
                     epochs=50)
    summary(best_model.model, (1, input_dim))
    best_model.fit(norm_train_set, norm_val_set, y_train_range, y_val_range)
    test_pred = best_model.predict(norm_test_set, y_test_range)
