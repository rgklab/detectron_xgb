import numpy as np
from numpy import ndarray
from sklearn.model_selection import train_test_split


class EarlyStopper:
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.wait = 0
        assert mode in ['min', 'max']
        self.mode = mode

    def update(self, metric):
        if self.best is None:
            self.best = metric
            return False
        if self.mode == 'min':
            if metric < self.best - self.min_delta:
                self.best = metric
                self.wait = 0
                return False
            else:
                self.wait += 1
                return self.wait >= self.patience
        elif self.mode == 'max':
            if metric > self.best + self.min_delta:
                self.best = metric
                self.wait = 0
                return False
            else:
                self.wait += 1
                return self.wait >= self.patience


def train_test_val_split(data, labels, sizes=(7, 1, 2), random_state=0, verbose=False) \
        -> dict[str, tuple[ndarray, ndarray]]:
    sizes = np.array(sizes)
    train_size, test_size, val_size = sizes / sizes.sum()

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, random_state=random_state)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_size / (1 - test_size),
                                                      random_state=random_state)
    if verbose:
        print(
            f"Train size: {train_size:.2f} ({len(x_train)}), "
            f"Test size: {test_size:.2f} ({len(x_test)}), "
            f"Val size: {val_size:.2f} ({len(x_val)})"
        )
    return dict(train=(x_train, y_train), test=(x_test, y_test), val=(x_val, y_val))
