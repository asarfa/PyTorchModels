from typing import List, Tuple, Optional
import time
import os


import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from asfarapi.utils import EarlyStopping


class Trainer:
    """
    This class allows to compute the main methods to fit and evaluate a model and predict output on testing set
    """

    def __init__(
        self,
        model,
        epochs: int = 10,
        lr: float = 0.01,
        opt: str = 'SGD',
        seed: int = 42,
        verbose: bool = True,
        criterion: nn.Module = nn.CrossEntropyLoss(),
        eval_criterion: Optional[nn.Module] = None,
        save_path: str = "./models",
        transforms: List[object] = []
    ):
        self.seed = seed
        self.device = torch.device(Trainer.get_current_device())
        self.criterion = criterion
        self.lr = lr
        self.epochs = epochs
        self.transforms = transforms
        self.opt = opt
        self.verbose = verbose
        self.shuffle = True
        if eval_criterion:
            self.eval_criterion = eval_criterion
        else:
            self.eval_criterion = criterion
        self.train_loss = None
        self.val_loss = None
        self.best_epoch = None
        self.base_save_path = save_path
        self.__set_seed()
        self.__instantiate_model(model)
        self.__instantiate_optimizer()

    @staticmethod
    def get_current_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return 'mps'
        return "cpu"

    def to_device(self, data) -> torch.Tensor:
        if isinstance(data, (list, tuple)):
            return [self.to_device(x) for x in data]
        return data.to(self.device, non_blocking=True)

    def __set_seed(self) -> None:
        torch.manual_seed(self.seed)

    def __instantiate_model(self, model: torch.nn.Module) -> None:
        result = self.to_device(model)
        if isinstance(result, torch.nn.Module):
            self.model = result
        else:
            print("Error is due to model not being a Module torch")
            raise Exception

    def __instantiate_optimizer(self) -> None:
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
    def compute_argmax(pred: torch.Tensor) -> torch.Tensor:
        pred = torch.argmax(torch.exp(torch.log_softmax(pred, dim=1)), dim=1)
        return pred

    def accuracy(self, pred: torch.Tensor, true: torch.Tensor) -> float:
        pred = self.compute_argmax(pred)
        correct = torch.eq(pred.cpu(), true.cpu()).int()
        return float(correct.sum()) / float(correct.numel())

    def __train_model(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        # set the model in training mode
        self.model.train()
        # stores the loss
        train_losses, train_accuracy = [], []
        if self.verbose:
            generator = tqdm(
                dataloader, leave=True, desc="Training on the epoch..."
            )
        else:
            generator = dataloader
        # losses = []
        for X, y in generator:
            # send input to device
            if self.transforms != []:
                for t in self.transforms:
                    X, y = t(X, y)
            X, y = self.to_device((X, y))
            # zero out previous accumulated gradients
            self.optimizer.zero_grad()
            # perform forward pass and calculate accuracy + loss
            outputs = self.model(X)
            loss = self.criterion(outputs, y)
            train_losses.append(loss.item())
            if self.verbose:
                generator.set_description(
                    f"Loss is: {round(np.mean(train_losses), 3)}")
            # perform backpropagation and update model parameters
            loss.backward()

            self.optimizer.step()

            train_accuracy.append(self.accuracy(outputs, y))

        return np.mean(train_losses), np.mean(train_accuracy)

    @torch.no_grad()
    def __evaluate_model(self, dataloader: DataLoader):
        # Allows to evaluate on dataloader or predict on datalaoder
        # set the model in eval mode
        self.model.eval()
        losses, accuracy, predictions = [], [], []
        if self.verbose:
            generator = tqdm(
                dataloader, leave=True, desc="Evaluating model...")
        else:
            generator = dataloader
        for X, y in generator:
            X, y = self.to_device((X, y))
            # perform forward pass and calculate accuracy + loss
            outputs = self.model(X)
            loss = self.criterion(outputs, y)
            losses.append(loss.item())
            if self.verbose:
                generator.set_description(
                    f"val loss: {round(np.mean(losses), 3)}")

            predictions.append(self.compute_argmax(outputs))
            accuracy.append(self.accuracy(outputs, y))
        return np.mean(losses), np.mean(accuracy), predictions

    def __compute_early_stopping(self, epoch: int, my_es: EarlyStopping, val_loss_mean: np.ndarray) -> bool:
        break_it = False
        my_es(val_loss_mean)
        if my_es.early_stop:
            if self.verbose:
                print(
                    f'At last epoch {epoch}, the second early stopping tolerance = {my_es.tolerance} has been reached,'
                    f' the loss of validation is not decreasing anymore -> stop it')
            break_it = True
        return break_it

    def __compute_verbose_train(
        self,
        epoch: int,
        start_time: float,
        train_loss_mean: np.ndarray,
        val_loss_mean: np.ndarray,
        train_acc_mean: np.ndarray,
        val_acc_mean: np.ndarray
    ) -> None:
        print(
            f"""Epoch [{epoch}] took {time.time() - start_time:.2f}s | \
                train_loss: {train_loss_mean:.4f}, \
                train_acc: {train_acc_mean:.4f}, \
                val_loss: {val_loss_mean:.4f}, \
                val_acc: {val_acc_mean:.4f}"""
        )

    def update_lr(self, epoch: int) -> None:
        if epoch == 30:
            self.lr *= 0.8
        if epoch == 25:
            self.lr *= 0.8

    def fit(self, dataset_train: DataLoader, dataset_val: DataLoader) -> None:
        # Code to update over here lots of possibly unbounds
        my_es = EarlyStopping(tolerance=20)
        generator = range(1, self.epochs + 1)
        if self.verbose:
            print("Starting the trainning ... ")

        for epoch in generator:
            start_time = time.time()
            train_loss_mean, train_acc_mean = self.__train_model(dataset_train)
            val_loss_mean, val_acc_mean, _ = self.__evaluate_model(dataset_val)
            if self.verbose:
                print(
                    f"Epoch {epoch} finished with train_loss: {train_loss_mean}, and val_loss: {val_loss_mean}")

            break_it = self.__compute_early_stopping(
                epoch, my_es, val_loss_mean)
            # self.update_lr(epoch)
            if break_it:
                break

            # to be able to restore the best weights with the best epoch
            torch.save(self.model.state_dict(), os.path.join(
                self.base_save_path, f'model_{epoch}.pt'))

        if self.verbose:
            self.__compute_verbose_train(epoch, start_time, train_loss_mean, val_loss_mean, train_acc_mean,
                                         val_acc_mean)

        if break_it:
            self.best_epoch = epoch - my_es.tolerance
        else:
            self.best_epoch = epoch
        self.train_acc = train_acc_mean
        self.val_acc = val_acc_mean

    def load_model(self) -> None:
        path = os.path.join(self.base_save_path, f'model_{self.best_epoch}.pt')
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def predict(self, dataset: DataLoader) -> torch.Tensor:
        self.shuffle = False
        self.load_model()
        loss_mean, acc_mean, predictions = self.__evaluate_model(dataset)
        print(
            f'The testing loss is {loss_mean}, the testing acc is {acc_mean}'
        )
        return predictions
