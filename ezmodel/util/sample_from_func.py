import numpy as np


def create(func, n_train, n_test, n_var=1, seed=1):
    np.random.seed(seed)
    X = np.random.random((n_train, n_var)) * 2 * np.pi - np.pi
    y = func(X).sum(axis=1)[:, None]

    _X = np.random.random((n_test, n_var)) * 2 * np.pi - np.pi
    _y = func(_X).sum(axis=1)[:, None]

    return X, y, _X, _y


def sine_function(n_train, n_test):
    return create(lambda X: np.sin(5*X), n_train, n_test)


def constant_function(n_train, n_test, noise=0.0, **kwargs):
    return create(lambda X: np.random.randn(len(X))[:, None] * noise, n_train, n_test)


def square_function(n_train, n_test, noise=0.0, **kwargs):
    return create(lambda X: X ** 2 + np.random.randn(len(X))[:, None] * noise, n_train, n_test)


def linear_function(n_train, n_test, noise=0.0, **kwargs):
    return create(lambda X: X + np.random.randn(len(X))[:, None] * noise, n_train, n_test, **kwargs)
