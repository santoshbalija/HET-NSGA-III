import numpy as np

from ezmodel.core.transformation import Transformation


class ZeroToOneNormalization(Transformation):

    def __init__(self, xl=None, xu=None, estimate_bounds=True) -> None:
        super().__init__()
        self.xl = xl
        self.xu = xu
        self.estimate_bounds = estimate_bounds

    def forward(self, X):

        if self.estimate_bounds:
            if self.xl is None:
                self.xl = np.min(X, axis=0)
            if self.xu is None:
                self.xu = np.max(X, axis=0)

        xl, xu = self.xl, self.xu

        # if np.any(xl == xu):
        #     raise Exception("Normalization failed because lower and upper bounds are equal!")

        # calculate the denominator
        denom = xu - xl

        # we can not divide by zero -> plus small epsilon
        denom += (denom == 0) * 1e-32

        # normalize the actual values
        N = (X - xl) / denom

        return N

    def backward(self, X):
        return X * (self.xu - self.xl) + self.xl
