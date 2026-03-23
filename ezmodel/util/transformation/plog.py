import numpy as np

from ezmodel.core.transformation import Transformation


class Plog(Transformation):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, y):
        yp = np.zeros_like(y)

        larger = y >= 0
        yp[larger] = np.log(1 + y[larger])
        yp[~larger] = - np.log(1 - y[~larger])

        return yp

    def backward(self, yp):
        y = np.zeros_like(yp)

        larger = yp >= 0
        y[larger] = np.exp(yp[larger]) - 1
        y[~larger] = 1 - np.exp(-yp[~larger])

        return y
