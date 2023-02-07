import time

import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.early_stoping import EarlyStopping


class Trainer:
    """
    This class allows to compute the main methods to fit and evaluate a model and predict output on testing set
    """

    def __init__(self, model, epochs: int = 10, lr: float = 0.01, opt: str = 'SGD',
                 batch_size: int = 10, seed: int = 42, verbose: bool = True,
                 criterion=nn.CrossEntropyLoss()):
        self.seed = seed
        self.device = torch.device(Trainer.get_current_device())
        self.criterion = criterion
        self.lr = lr
        self.epochs = epochs
        self.opt = opt
        self.batch_size = batch_size
        self.verbose = verbose
        self.shuffle = True
        self.train_loss = None
        self.val_loss = None
        self.best_epoch = None
        self.__set_seed()
        self.__instantiate_model(model)
        self.__instantiate_optimizer()

    @staticmethod
    def get_current_device():
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return 'mps'
        return "cpu"

    def to_device(self, data):
        if isinstance(data, (list, tuple)):
            return [self.to_device(x) for x in data]
        return data.to(self.device, non_blocking=True)

    def __set_seed(self):
        torch.manual_seed(self.seed)

    def __instantiate_model(self, model: torch.nn.Module):
        result = self.to_device(model)
        if isinstance(result, torch.nn.Module):
            self.model = result
        else:
            print("Error is due to model not being a Module torch")
            raise Exception

    def __instantiate_optimizer(self):
        if self.opt == 'Adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.lr)
        elif self.opt == 'SGD':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.lr)
        elif self.opt == 'RMS':
            self.optimizer = torch.optim.RMSprop(
                self.model.parameters(), lr=self.lr)
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
            loss = self.criterion(outputs, y.long())
            # perform backpropagation and update model parameters
            loss.backward()
            self.optimizer.step()

            train_losses.append(loss.item())
            train_accuracy.append(self.accuracy(outputs, y))

        return np.mean(train_losses), np.mean(train_accuracy)

    @torch.no_grad()
    def __evaluate_model(self, dataset):
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
        # Code to update over here lots of possibly unbounds
        my_es = EarlyStopping()

        for epoch in range(1, self.epochs + 1):
            start_time = time.time()
            train_loss_mean, train_acc_mean = self.__train_model(dataset_train)
            val_loss_mean, val_acc_mean, _ = self.__evaluate_model(dataset_val)

            break_it = self.__compute_early_stopping(
                epoch, my_es, val_loss_mean)
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
        print(
            f'The testing loss is {loss_mean}, the testing acc is {acc_mean}')
        return predictions
