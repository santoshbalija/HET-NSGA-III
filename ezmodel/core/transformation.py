from abc import abstractmethod


class Transformation:

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, X):
        pass

    @abstractmethod
    def backward(self, X):
        pass


class NoNormalization(Transformation):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, X):
        return X

    def backward(self, X):
        return X
