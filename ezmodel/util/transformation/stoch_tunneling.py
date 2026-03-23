from ezmodel.core.transformation import Transformation
import numpy as np


class StochasticTunneling(Transformation):

    def __init__(self, gamma, y_min=None) -> None:
        super().__init__()
        self.y_min = y_min
        self.gamma = gamma

    def forward(self, y):
        if self.y_min is None:
            self.y_min = y.min(axis=0)

        return 1 - np.exp(- self.gamma * (y - self.y_min))

    def backward(self, yp):
        yp = np.clip(yp, None, 1.0 - 1e-16)
        return self.y_min - np.log(1 - yp) / self.gamma


if __name__ == "__main__":
    F = np.random.random((100, 1))

    tunneling = StochasticTunneling(0.1)
    Fp = tunneling.forward(F)
