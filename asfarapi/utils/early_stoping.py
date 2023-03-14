import numpy as np


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
