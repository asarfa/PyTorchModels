import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchsummary import summary
import time
import numpy as np
import random
import pandas as pd


class Model(nn.Module):
    """
    This class allows to compute a multi-classification architecture based on Conv2d and hyper-parameters
    such as the number of neurons, the number of hidden layers, the kernel size, the activation
    of hidde layers
    """

    def __init__(self, input_shape: tuple, output_dim: int, params: dict, last_activation: bool):
        super(Model, self).__init__()
        params['input_dim'] = 1
        params['output_dim'] = output_dim
        self.stack_layers = self.get_layers(params)
        block_output_shape = self.compute_output_block_shape(params['kernel_size'], params['n_hidden'], input_shape)
        self.fcl = nn.Linear(block_output_shape[0] * block_output_shape[1] * params['hidden_dim'], params['output_dim'])
        self.last_activation = last_activation
        self.compute_set()

    @staticmethod
    def conv2d(params):
        return nn.Conv2d(in_channels=params['input_dim'], out_channels=params['hidden_dim'],
                         kernel_size=params['kernel_size'], padding=0, dilation=1, stride=1)

    @staticmethod
    def linear(input_dim, output_dim):
        return nn.Linear(in_features=input_dim, out_features=output_dim)

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
    def flatten():
        return nn.Flatten()

    @staticmethod
    def softmax():
        return nn.Softmax()

    @staticmethod
    def set_conv2d(stack_layers: list, params):
        stack_layers.append(Model.conv2d(params))
        if params['activation'] == 'relu':
            act = Model.relu()
        elif params['activation'] == 'sigmoid':
            act = Model.sigmoid()
        elif params['activation'] == 'tanh':
            act = Model.tanh()
        stack_layers.append(act)
        return stack_layers

    @staticmethod
    def get_layers(params):
        stack_layers = []
        stack_layers = Model.set_conv2d(stack_layers, params)
        for i in range(params['n_hidden']):
            params['input_dim'] = params['hidden_dim']
            stack_layers = Model.set_conv2d(stack_layers, params)
        return stack_layers

    @staticmethod
    def compute_output_block_shape(kernel_size, n_layers, input_shape, padding=0, dilatation=1, stride=1):
        current_shape = [input_shape[0], input_shape[1]]
        for _ in range(n_layers + 1):
            current_shape[0] = ((current_shape[0] + 2 * padding - dilatation * (kernel_size - 1) - 1) / stride) + 1
            current_shape[1] = ((current_shape[1] + 2 * padding - dilatation * (kernel_size - 1) - 1) / stride) + 1
        current_shape = [int(item) for item in current_shape]
        return current_shape

    def compute_set(self):
        for i, layer in enumerate(self.stack_layers):
            name = f'Layer_{str(i + 1).zfill(3)}'
            setattr(self, name, layer)

    def forward(self, x):
        for i, layer in enumerate(self.stack_layers):
            x = getattr(self, f'Layer_{str(i + 1).zfill(3)}')(x)
        x = torch.flatten(x, 1)
        x = self.fcl(x)
        x = F.softmax(x, dim=1) if self.last_activation else x
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


class MultiClass:
    """
    This class allows to compute the main methods to fit and evaluate a model and predict output on testing set
    """

    def __init__(self, model, epochs: int = 10, lr: float = 0.01, opt: str = 'SGD',
                 batch_size: int = 10, seed: int = 42, verbose: bool = True,
                 criterion=nn.CrossEntropyLoss()):
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

    def __instantiate_optimizer(self):
        if self.opt == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.opt == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        elif self.opt == 'RMS':
            self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.lr)
        else:
            raise ValueError(f'{self.optimizer} has not been implemented')

    @staticmethod
    def compute_argmax(pred):
        pred = torch.argmax(torch.exp(torch.log_softmax(pred, dim=1)), dim=1)
        return pred

    def accuracy(self, pred, true):
        pred = self.compute_argmax(pred)
        correct = torch.eq(pred.cpu(), true.cpu()).int()
        return float(correct.sum()) / float(correct.numel())

    def __train_model(self, dataset):
        # One hot encoding or not according to the loss
        one_hot = True if isinstance(self.criterion, nn.MSELoss) else False
        # Instantiate the dataloader from dataset
        dataloader = DataLoader(dataset, self.batch_size, shuffle=True)
        # set the model in training mode
        self.model.train()
        # stores the loss
        train_losses, train_accuracy = [], []
        for X, y in dataloader:
            # send input to device
            X, y = self.to_device((X, y))
            # zero out previous accumulated gradients
            self.optimizer.zero_grad()
            # perform forward pass and calculate accuracy + loss
            outputs = self.model(X)
            if one_hot:
                # One-hot encoding or labels so as to calculate MSE error:
                labels_one_hot = torch.FloatTensor(len(X), 10)
                labels_one_hot = labels_one_hot.to(self.device)
                labels_one_hot.zero_()
                labels_one_hot.scatter_(1, y.view(-1, 1), 1)
                loss = self.criterion(outputs, labels_one_hot)  # Real number
            else:
                loss = self.criterion(outputs, y.long())
            # perform backpropagation and update model parameters
            loss.backward()
            self.optimizer.step()

            train_losses.append(loss.item())
            train_accuracy.append(self.accuracy(outputs, y))

        return np.mean(train_losses), np.mean(train_accuracy)

    @torch.no_grad()
    def __evaluate_model(self, dataset):
        # One hot encoding or not according to the loss
        one_hot = True if isinstance(self.criterion, nn.MSELoss) else False
        # Instantiate the dataloader from dataset
        dataloader = DataLoader(dataset, self.batch_size, shuffle=self.shuffle)
        # Allows to evaluate on dataloader or predict on datalaoder 
        # set the model in eval mode
        self.model.eval()
        losses, accuracy, predictions = [], [], []
        for X, y in dataloader:
            X, y = self.to_device((X, y))
            # perform forward pass and calculate accuracy + loss
            outputs = self.model(X)
            if one_hot:
                # One-hot encoding or labels so as to calculate MSE error:
                labels_one_hot = torch.FloatTensor(len(X), 10)
                labels_one_hot = labels_one_hot.to(self.device)
                labels_one_hot.zero_()
                labels_one_hot.scatter_(1, y.view(-1, 1), 1)
                loss = self.criterion(outputs, labels_one_hot)  # Real number
            else:
                loss = self.criterion(outputs, y.long())

            predictions.append(self.compute_argmax(outputs))
            losses.append(loss.item())
            accuracy.append(self.accuracy(outputs, y))
        return np.mean(losses), np.mean(accuracy), predictions

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

    def __compute_verbose_train(self, epoch, start_time, train_loss_mean, val_loss_mean, train_acc_mean, val_acc_mean):
        print(
            "Epoch [{}] took {:.2f}s | train_loss: {:.4f}, train_acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
                epoch, time.time() - start_time, train_loss_mean, train_acc_mean, val_loss_mean, val_acc_mean))

    def fit(self, dataset_train, dataset_val):
        my_es = EarlyStopping()

        for epoch in range(1, self.epochs + 1):
            start_time = time.time()
            train_loss_mean, train_acc_mean = self.__train_model(dataset_train)
            val_loss_mean, val_acc_mean, _ = self.__evaluate_model(dataset_val)

            break_it = self.__compute_early_stopping(epoch, my_es, val_loss_mean)
            if break_it:
                break

            # to be able to restore the best weights with the best epoch
            torch.save(self.model.state_dict(), f'model_{epoch}.pt')

        if self.verbose:
            self.__compute_verbose_train(epoch, start_time, train_loss_mean, val_loss_mean, train_acc_mean,
                                         val_acc_mean)

        if break_it:
            self.best_epoch = epoch - my_es.tolerance
        else:
            self.best_epoch = epoch
        self.train_acc = train_acc_mean
        self.val_acc = val_acc_mean

    def load_model(self):
        path = f'model_{self.best_epoch}.pt'
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def predict(self, dataset):
        self.batch_size = 1
        self.shuffle = False
        self.load_model()
        loss_mean, acc_mean, predictions = self.__evaluate_model(dataset)
        print(f'The testing loss is {loss_mean}, the testing acc is {acc_mean}')
        return predictions


if __name__ == '__main__':
    """
    Loading dataset

    import torchvision
    import torchvision.transforms as transforms
    import requests
    import os
    url = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.bz2'
    r = requests.get(url, allow_redirects=True)
    if not os.path.isdir('USPS/'):
        os.mkdir('USPS/')
    open('USPS/usps.bz2', 'wb').write(r.content)

    dataset = torchvision.datasets.USPS(root='USPS/',
                                            train=True, 
                                            transform=transforms.ToTensor(),
                                            download=False)

    train_set, val_set = random_split(dataset, [6000, 1291])
    """
    train_set, val_set = None, None

    """
    defining space of hyper parameters
    """
    hidden_dim = [5, 10, 20]
    n_hidden = [1, 2, 3, 5]
    activation = ["sigmoid", "relu", "tanh"]
    kernel_size = [1, 2, 3]
    batch_size = [10, 100, 300, 500, 1000]
    learning_rate = [1e-3, 1e-2, 1e-1, 1, 1e1]
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
    model = Model(input_shape=input_shape, output_dim=output_dim, params=dict_param[0], last_activation=False)
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
                 'learning_rate', 'optimizer', 'best_epoch']
        , index=range(len(random_key)))
    best_val_acc = -np.inf
    best_params = None
    start = time.time()
    for i, key in enumerate(random_key):
        params = dict_param[key]
        params_architecture = dict(list(params.items())[:4])
        model = MultiClass(Model(input_shape=input_shape, output_dim=output_dim, params=params_architecture,
                                 last_activation=False),
                           verbose=False, lr=params['lr'],
                           opt=params['optimizer'], batch_size=params['batch_size'],
                           epochs=50, criterion=nn.CrossEntropyLoss())
        model.fit(train_set, val_set)
        train_acc = round(model.train_acc, 4)
        val_acc = round(model.val_acc, 4)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = params
        results.iloc[i] = [train_acc, val_acc] + list(params.values()) + [model.best_epoch]
    print(f'Time elapsed: {round((time.time() - start) / 60, 3)} minutes')
    results = results.sort_values('val_acc', ascending=False)
    print('Hyper parameters sorted according maximum validation accuracy: ')
    print(results)

    """
    Loading test set
    
    url = 'http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.t.bz2'
    r = requests.get(url, allow_redirects=True)
    open('USPS/usps.t.bz2', 'wb').write(r.content)
    test_set = torchvision.datasets.USPS(root='USPS/',
                                            train=False,
                                            transform=transforms.ToTensor(),
                                            download=False)
    """
    test_set = None
    """
    Predicting on test set with best hyper params set
    """
    print(f'Best hyper params: {best_params}')
    best_params_architecture = dict(list(best_params.items())[:4])
    best_model = MultiClass(Model(input_shape=input_shape, output_dim=output_dim, params=best_params_architecture,
                                  last_activation=False),
                            verbose=False, lr=best_params['lr'],
                            opt=best_params['optimizer'], batch_size=best_params['batch_size'],
                            epochs=50, criterion=nn.CrossEntropyLoss())
    summary(best_model.model, (1, input_shape[0], input_shape[1]))
    best_model.fit(train_set, val_set)
    test_pred = best_model.predict(test_set)
