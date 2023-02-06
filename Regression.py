import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchsummary import summary
import time
import numpy as np
import random
import pandas as pd
import math


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
        loss = torch.sum(math.log(2 * math.pi) + logvars + (targets - means) ** 2 / torch.exp(logvars))
        return torch.mean(1 / 2 * loss)


class Model(nn.Module):
    """
    This class allows to compute a regression architecture based on Linear and hyper-parameters
    such as the number of neurons, the number of hidden layers, the kernel size, the activation
    of hidde layers
    """

    def __init__(self, input_dim: int, output_dim: int, params: dict,
                 last_activation: bool = True):
        super(Model, self).__init__()
        hidden_dim: int = params['hidden_dim']
        n_hidden: int = params['n_hidden']
        activation: str = params['activation']
        self.stack_layers = self.get_layers(input_dim, hidden_dim, n_hidden, activation)
        self.fcl = nn.Linear(hidden_dim, output_dim)
        self.last_activation = last_activation  # prices are supposed to be >=0, hence relu
        self.compute_set()

    @staticmethod
    def linear(input_dim, hidden_dim):
        return nn.Linear(input_dim, hidden_dim)

    @staticmethod
    def relu():
        return nn.ReLU()

    @staticmethod
    def sigmoid():
        return nn.Sigmoid()

    @staticmethod
    def tanh():
        return nn.Tanh()

    @staticmethod
    def set_linear(stack_layers: list, input_dim, hidden_dim, activation: str):
        stack_layers.append(Model.linear(input_dim, hidden_dim))
        if activation == 'relu':
            act = Model.relu()
        elif activation == 'sigmoid':
            act = Model.sigmoid()
        elif activation == 'tanh':
            act = Model.tanh()
        stack_layers.append(act)
        return stack_layers

    @staticmethod
    def get_layers(input_dim: int, hidden_dim: int, n_hidden: int, activation: str):
        stack_layers = []
        stack_layers = Model.set_linear(stack_layers, input_dim, hidden_dim, activation)
        for i in range(n_hidden):
            input_dim = hidden_dim
            stack_layers = Model.set_linear(stack_layers, input_dim, hidden_dim, activation)
        return stack_layers

    def compute_set(self):
        for i, layer in enumerate(self.stack_layers):
            name = f'Layer_{str(i + 1).zfill(3)}'
            setattr(self, name, layer)

    def forward(self, x):
        for i, layer in enumerate(self.stack_layers):
            x = getattr(self, f'Layer_{str(i + 1).zfill(3)}')(x)
        x = self.fcl(x)
        x = self.relu()(x) if self.last_activation else x
        return x


class EarlyStopping:
    """
    This class allows to stop the training when the validation loss doesn't decrease anymore
    """

    def __init__(
            self,
            tolerance: int = 10):

        self.tolerance = tolerance
        self.counter = 0
        self.early_stop = False
        self.best_val_loss = np.inf

    def __call__(self, val_loss):
        if round(val_loss, 3) >= round(self.best_val_loss, 3):
            self.counter += 1
            if self.counter >= self.tolerance:
                self.early_stop = True
        else:
            self.best_val_loss = val_loss
            self.counter = 0


class Reg:
    """
    This class allows to compute the main methods to fit a model and predict output on testing set
    """

    def __init__(self, model, epochs: int = 10, lr: float = 0.01, opt: str = 'SGD',
                 batch_size: int = 10, seed: int = 42, verbose: bool = True,
                 criterion=nn.MSELoss()):
        self.seed = seed
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.criterion = criterion
        self.lr = lr
        self.epochs = epochs
        self.opt = opt
        self.batch_size = batch_size
        self.verbose = verbose
        self.shuffle = True
        self.optimizer = None
        self.train_loss = None
        self.val_loss = None
        self.best_epoch = None
        self.__set_seed()
        self.__instantiate_model(model)
        self.__instantiate_optimizer()

    def to_device(self, data):
        if isinstance(data, (list, tuple)):
            return [self.to_device(x) for x in data]
        return data.to(self.device, non_blocking=True)

    def __set_seed(self):
        torch.manual_seed(self.seed)

    def __instantiate_model(self, model):
        self.model = self.to_device(model)

    @staticmethod
    def rmse(pred, true):
        pred = pred.detach().cpu().numpy()
        true = true.detach().cpu().numpy()
        return np.sqrt(np.sum((pred - true) ** 2) / len(true))

    def __instantiate_optimizer(self):
        if self.opt == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.opt == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        elif self.opt == 'RMS':
            self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.lr)
        else:
            raise ValueError(f'{self.optimizer} has not been implemented')

    def __train_model(self, dataset: TensorDataset, y_range=None):
        # Instantiate the dataloader from dataset
        dataloader = DataLoader(dataset, self.batch_size, shuffle=True)
        # set the model in training mode
        self.model.train()
        # stores the loss
        train_losses, train_rmse = [], []
        for X, y in dataloader:
            # send input to device
            # X, y = self.__transf_batch(X, y)
            X, y = self.to_device((X, y))
            # zero out previous accumulated gradients
            self.optimizer.zero_grad()
            # perform forward pass and calculate accuracy + loss
            outputs = self.model(X)
            loss = self.criterion(outputs, y.unsqueeze(1))
            # perform backpropagation and update model parameters
            loss.backward()
            self.optimizer.step()
            if isinstance(self.criterion, nn.MSELoss):
                outputs = undo_scale(outputs, y_range)
                y = undo_scale(y, y_range)
                true_loss = self.criterion(outputs, y.unsqueeze(1))
                train_rmse.append(self.rmse(outputs, y))
            else:  # for Gaussian Likelihood loss, target is the first column vector of outputs
                true_loss = loss
                target = undo_scale(outputs[:, 0], y_range)
                train_rmse.append(self.rmse(target, y))
            train_losses.append(true_loss.item())
        return np.mean(train_losses), np.mean(train_rmse)

    @torch.no_grad()
    def __evaluate_model(self, dataset: TensorDataset, y_range=None):
        # Instantiate the dataloader from dataset
        dataloader = DataLoader(dataset, self.batch_size, shuffle=self.shuffle)
        # Allows to evaluate on dataloader or predict on datalaoder 
        # set the model in eval mode
        self.model.eval()
        losses, rmse, predictions = [], [], []
        for X, y in dataloader:
            # X, y = self.__transf_batch(X, y)
            X, y = self.to_device((X, y))
            # perform forward pass and calculate accuracy + loss
            outputs = self.model(X)
            y = undo_scale(y, y_range)
            if isinstance(self.criterion, nn.MSELoss):
                outputs = undo_scale(outputs, y_range)
                rmse.append(self.rmse(outputs, y))
            else:  # for Gaussian Likelihood loss, target is the first column vector of outputs
                target = undo_scale(outputs[:, 0], y_range)
                rmse.append(self.rmse(target, y))
            predictions.append(outputs)
            loss = self.criterion(outputs, y.unsqueeze(1))
            losses.append(loss.item())
        return np.mean(losses), np.mean(rmse), predictions

    def __compute_early_stopping(self, epoch, my_es, val_loss_mean):
        break_it = False
        my_es(val_loss_mean)
        if my_es.early_stop:
            if self.verbose:
                print(
                    f'At last epoch {epoch}, the second early stopping tolerance = {my_es.tolerance} has been reached,'
                    f' the loss of validation is not decreasing anymore -> stop it')
            break_it = True
        return break_it

    def __compute_verbose_train(self, epoch, start_time, train_loss_mean, val_loss_mean, train_rmse_mean,
                                val_rmse_mean):
        print(
            "Epoch [{}] took {:.2f}s | train_loss: {:.4f}, train_rmse: {:.4f}, val_loss: {:.4f}, val_rmse: {:.4f}".format(
                epoch, time.time() - start_time, train_loss_mean, train_rmse_mean, val_loss_mean, val_rmse_mean))

    def fit(self, dataset_train: TensorDataset, dataset_val: TensorDataset,
            y_train_range=None, y_val_range=None):
        my_es = EarlyStopping()

        for epoch in range(1, self.epochs + 1):
            start_time = time.time()
            train_loss_mean, train_rmse_mean = self.__train_model(dataset_train, y_train_range)
            val_loss_mean, val_rmse_mean, _ = self.__evaluate_model(dataset_val, y_val_range)

            break_it = self.__compute_early_stopping(epoch, my_es, val_loss_mean)
            if break_it:
                break

            # to be able to restore the best weights with the best epoch
            torch.save(self.model.state_dict(), f'model_{epoch}.pt')

        if self.verbose: self.__compute_verbose_train(epoch, start_time, train_loss_mean, val_loss_mean,
                                                      train_rmse_mean, val_rmse_mean)

        if break_it:
            self.best_epoch = epoch - my_es.tolerance
        else:
            self.best_epoch = epoch
        self.train_rmse = train_rmse_mean
        self.val_rmse = val_rmse_mean

    def load_model(self):
        path = f'model_{self.best_epoch}.pt'
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def predict(self, dataset: TensorDataset, y_range=None):
        self.batch_size = 1
        self.shuffle = False
        self.load_model()
        loss_mean, rmse_mean, predictions = self.__evaluate_model(dataset, y_range)
        print(f'The testing RMSE is {rmse_mean}')
        return predictions


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
                 'optimizer', 'best_epoch']
        , index=range(len(random_key)))
    best_val_rmse = np.inf
    best_params = None
    start = time.time()
    for i, key in enumerate(random_key):
        params = dict_param[key]
        params_architecture = dict(list(params.items())[:3])
        model = Reg(Model(input_dim=input_dim, output_dim=output_dim, params=params_architecture),
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
        results.iloc[i] = [train_rmse, val_rmse] + list(params.values()) + [model.best_epoch]
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
