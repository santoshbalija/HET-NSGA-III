import numpy as np

from ezmodel.core.model import Model
from ezmodel.util.dist import calc_dist


class RBF(Model):

    def __init__(self,
                 kernel="linear",
                 tail="constant",
                 sigma=1.0,
                 rho=1e-6,
                 normalized=False,
                 optimize=False,
                 **kwargs) -> None:

        super().__init__(eliminate_duplicates=True, eliminate_duplicates_eps=1e-8, **kwargs)
        self.tail = tail
        self.rho = rho
        self.sigma = sigma
        self.normalized = normalized
        self.optimize = optimize

        if kernel == "linear":
            kernel = kernel_linear
        elif kernel == "quadratic":
            kernel = kernel_quadratic
        elif kernel == "cubic":
            kernel = kernel_cubic
        elif kernel == "gaussian":
            kernel = kernel_gaussian
        elif kernel == "mq":
            kernel = kernel_multi_quadr
        elif kernel == "tps":
            kernel = kernel_tps
        elif kernel == "periodic":
            kernel = kernel_periodic
        else:
            raise Exception("Unknown Kernel Function: " + kernel)

        self.kernel = kernel

    def _fit(self, X, y, **kwargs):

        rho, tail, kernel, sigma, normalized = self.rho, self.tail, self.kernel, self.sigma, self.normalized

        if self.optimize:

            sigmas = np.linspace(0.0001, 20, 30)
            models = [rbf_fit(X, y, kernel, sigma=sigma, tail=tail, rho=rho, normalized=normalized) for sigma in sigmas]
            f = np.array([model["loocv"] for model in models])
            cond = np.array([model["cond"] for model in models])
            f[cond > 1e12] = np.inf

            # _f = np.array([calc_loocv(X, y, kernel, sigma=sigma, tail=tail, rho=rho)[0] for sigma in sigmas])
            # assert np.all(np.logical_or(cond > 1e12, np.abs(f - _f) < 1e-6))

            k = f.argmin()

            print(sigmas[k])

            self.model = models[k]

            # print(np.all(np.logical_or(cond > 1e12, np.abs(f - _f) < 1e-6)))

        else:
            self.model = rbf_fit(X, y, kernel, tail=tail, rho=rho, sigma=sigma, normalized=normalized)

    def _predict(self, X, out):
        y_hat = rbf_predict(self.model, X)
        out["y"] = y_hat

    @classmethod
    def hyperparameters(cls):
        return {
            "kernel": ["linear", "cubic", "gaussian", "mq"],
            "tail": ["constant", "linear", "quadratic", "linear+quadratic"],
            # "rho": [None, 1e-3, 1e-2, 1e-1, 0.5, 1.0, 2.0],
            "normalized": [True, False],
        }


def rbf_kernel(X, phi, tail="linear", **kwargs):
    n, m = X.shape

    if tail is None:
        P = np.zeros((n, 0))
    if tail == "constant":
        P = np.ones((n, 1))
    elif tail == "linear":
        P = np.column_stack((np.ones(n), X))
    elif tail == "quadratic":
        P = np.column_stack((np.ones(n), X ** 2))
    elif tail == "linear+quadratic":
        P = np.column_stack((np.ones(n), X, X ** 2))

    return np.column_stack([phi, P])


def rbf_fit(X, y, func, Xp=None, rho=0.0, normalized=False, **kwargs):
    if Xp is None:
        Xp = X

    phi = calc_dist(X, Xp)
    phi = func(phi, **kwargs)

    phi_norm = 1.0
    if normalized:
        phi_norm = phi.sum(axis=0)[None, :]
        phi = phi / phi_norm

    if rho is not None:
        phi = phi + np.eye(len(phi)) * (rho ** 2)

    K = rbf_kernel(X, phi, **kwargs)
    n, m = K.shape

    lhs = np.zeros((m, m))
    lhs[:n, :m] = K
    lhs[n:, :n] = K[:, n:].T

    rhs = np.zeros((m, 1))
    rhs[:n] = y

    A_inv, cond = svd_inv(lhs)
    coef = A_inv @ rhs

    c, K = coef[:n, 0], A_inv[:n, :n]
    e = c / (np.diag(K) + 1e-128)
    loocv = (e ** 2).sum()
    gcv = (c ** 2).sum() / (np.diag(K).mean() ** 2)

    return dict(X=X, cond=cond, e=e, loocv=loocv, gcv=gcv, coef=coef, func=func, phi_norm=phi_norm, kwargs=kwargs)


def rbf_predict(model, X):
    phi = model["func"](calc_dist(X, model["X"]), **model["kwargs"])
    phi = phi / model["phi_norm"]
    phi = rbf_kernel(X, phi, **model["kwargs"])

    y = phi @ model["coef"]
    return y


def kernel_linear(r, sigma=1.0, **kwargs):
    return sigma * r


def kernel_quadratic(r, sigma=1.0, **kwargs):
    return (sigma * r) ** 2


def kernel_cubic(r, sigma=1.0, **kwargs):
    return (sigma * r) ** 3


def kernel_gaussian(r, sigma=None, **kwargs):
    return np.exp(- (sigma * r ** 2))


def kernel_periodic(r, sigma=1.0, **kwargs):
    return (sigma ** 2) * np.exp(- 2 * np.sin((np.pi * r) / 5) ** 2)


def kernel_multi_quadr(r, sigma=1.0, **kwargs):
    return ((r ** 2) + (sigma ** 2)) ** 0.5


def kernel_tps(r, **kwargs):
    r[r < np.finfo(float).eps] = np.finfo(float).eps
    return (r ** 2) * np.log(r)


def svd_inv(A):
    U, S, V = np.linalg.svd(A)
    Ainv = V.T @ np.diag(1 / S) @ U.T
    cond = np.abs(np.max(S)) / np.abs(np.min(S))
    return Ainv, cond


def calc_loocv(X, y, *args, **kwargs):
    e = []

    for k in range(len(X)):
        trn = [i for i in range(len(X)) if i != k]
        tst = [k]

        model = rbf_fit(X[trn], y[trn], *args, **kwargs)
        y_hat = rbf_predict(model, X[tst])[:, 0]
        y_true = y[tst, 0]

        e.append(y_true - y_hat)

    e = np.array(e)

    return (e ** 2).sum(), e
