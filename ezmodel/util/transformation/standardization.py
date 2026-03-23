import numpy as np

from ezmodel.core.transformation import Transformation


class Standardization(Transformation):

    def __init__(self, mean=None, std=None) -> None:
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, X):
        if self.mean is None:
            self.mean = np.mean(X, axis=0)

        if self.std is None:
            self.std = np.std(X, axis=0)

        val = (X - self.mean) / self.std
        return val

    def backward(self, X):
        return (X * self.std) + self.mean

