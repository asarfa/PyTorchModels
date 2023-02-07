import math

import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import TensorDataset


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
