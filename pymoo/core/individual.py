import copy

import numpy as np


class Individual:

    def __init__(self,
                 X: np.ndarray = [],
                 F: np.ndarray = [],
                 G: np.ndarray = [],
                 H: np.ndarray = [],
                 CV: np.ndarray = [],
                 dF: np.ndarray = [],
                 dG: np.ndarray = [],
                 dH: np.ndarray = [],
                 ddF: np.ndarray = [],
                 ddG: np.ndarray = [],
                 ddH: np.ndarray = [],
                 evaluated=None,
                 **kwargs) -> None:

        # design variables
        self._X = X

        # objectives and constraint values
        self._F = F
        self._G = G
        self._H = H

        # first order derivation
        self._dF = dF
        self._dG = dG
        self._dH = dH

        # second order derivation
        self._ddF = ddF
        self._ddG = ddG
        self._ddH = ddH

        # if the constraint violation value to be used
        self._CV = CV

        # a set storing what has been evaluated
        if evaluated is None:
            evaluated = set()
        self.evaluated = evaluated

        # additional data to be set
        self.data = kwargs

    def has(self, key, empty_list_as_none=True):
        if key in self.__dict__:
            v = self.__dict__[key]
            return v is not None or (empty_list_as_none and v != [])
        elif key in self.data:
            v = self.data[key]
            return v is not None or (empty_list_as_none and v != [])
        else:
            return False

    # -------------------------------------------------------
    # Values
    # -------------------------------------------------------

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, value):
        self._X = value

    @property
    def F(self):
        return self._F

    @F.setter
    def F(self, value):
        self._F = value

    @property
    def G(self):
        return self._G

    @G.setter
    def G(self, value):
        self._G = value

    @property
    def H(self):
        return self._H

    @H.setter
    def H(self, value):
        self._H = value

    @property
    def CV(self):
        return self._CV

    @CV.setter
    def CV(self, value):
        self._CV = value

    @property
    def FEAS(self):
        return self.CV <= 0.0

    # -------------------------------------------------------
    # Gradients
    # -------------------------------------------------------

    @property
    def dF(self):
        return self._dF

    @property
    def dG(self):
        return self._dG

    @property
    def dH(self):
        return self._dH

    # -------------------------------------------------------
    # Hessians
    # -------------------------------------------------------

    @property
    def ddF(self):
        return self._ddF

    @property
    def ddG(self):
        return self._ddG

    @property
    def ddH(self):
        return self._ddH

    # -------------------------------------------------------
    # Convenience (value instead of array)
    # -------------------------------------------------------

    @property
    def x(self):
        return self.X

    @property
    def f(self):
        return self.F[0]

    @property
    def cv(self):
        return self.CV[0]

    @property
    def feas(self):
        return self.FEAS[0]

    # -------------------------------------------------------
    # Deprecated
    # -------------------------------------------------------

    @property
    def feasible(self):
        return self.FEAS

    # -------------------------------------------------------
    # Other Functions
    # -------------------------------------------------------

    def set_by_dict(self, **kwargs):
        for k, v in kwargs.items():
            self.set(k, v)

    def set(self, key, value):
        if key in self.__dict__:
            self.__dict__[key] = value
        elif hasattr(self.__class__, key):
            setattr(self, key, value)
        else:
            self.data[key] = value
        return self

    def get(self, *keys):
        ret = []

        for key in keys:
            if key in self.__dict__:
                v = self.__dict__[key]
            elif hasattr(self.__class__, key):
                v = getattr(self, key)
            elif key in self.data:
                v = self.data[key]
            else:
                v = None

            ret.append(v)

        if len(ret) == 1:
            return ret[0]
        else:
            return tuple(ret)

    def duplicate(self, key, new_key):
        self.set(new_key, self.get(key))

    def new(self):
        return self.__class__()

    def copy(self, other=None, deep=True):
        obj = self.new()

        # if not provided just copy yourself
        if other is None:
            other = self

        # the data the new object needs to have
        D = other.__dict__

        # if it should be a deep copy do it
        if deep:
            D = copy.deepcopy(D)

        for k, v in D.items():
            obj.__dict__[k] = v

        return obj

