"""
Code from Sukrit Mittal
1. I've skipped MaF6, since it is the same as DTLZ5.
2. I've skipped MaF10/11/12 too, since they are WFG1, WFG2 and WFG9 respectively.
3. Don't use MaF15 from my code. I have coded it but there is some bug. If you can resolve that, it would be great.
    I used to generally skip it in my papers saying it is a large-scale problem (which it is).

"""

import math

import autograd.numpy as anp
import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.factory import get_problem
from scipy.spatial.distance import euclidean, cdist


class MaF1(ElementwiseProblem):

    def __init__(self, n_var=7, n_obj=3):
        xl = 0 * anp.ones(n_var)
        xu = 1 * anp.ones(n_var)

        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=0, xl=xl, xu=xu)

        # store custom variables needed for evaluation
        self.n_obj = n_obj
        self.n_var = n_var

    def _evaluate(self, x, out, *args, **kwargs):
        M = self.n_obj
        g = sum([(x[i] - 0.5) ** 2 for i in range(M - 1, len(x))])
        f = []
        for i in range(M):
            m = i + 1
            if m == 1:
                prod = np.prod(x[:M - m])
                f.append((1 - prod) * (1 + g))
            elif m < M:
                prod = np.prod(x[:M - m]) * (1 - x[M - m])
                f.append((1 - prod) * (1 + g))
            else:
                f.append(x[0] * (1 + g))
        out["F"] = f


class MaF2(ElementwiseProblem):

    def __init__(self, n_var=7, n_obj=3):
        xl = 0 * anp.ones(n_var)
        xu = 1 * anp.ones(n_var)

        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=0, xl=xl, xu=xu, elementwise_evaluation=True)

        # store custom variables needed for evaluation
        self.n_obj = n_obj
        self.n_var = n_var

    def _evaluate(self, x, out, *args, **kwargs):
        M = self.n_obj
        D = self.n_var
        g = []
        for m in range(M - 1):
            i = m + 1
            sum_val = 0
            for j in range(int(M + (i - 1) * np.floor((D - M + 1) / M)), int(M + i * np.floor((D - M + 1) / M))):
                sum_val += ((x[j - 1] / 2 + 1 / 4) - 0.5) ** 2
            g.append(sum_val)
        for m in range(M - 1, M):
            i = m + 1
            sum_val = 0
            for j in range(int(M + (i - 1) * np.floor((D - M + 1) / M)), D):
                sum_val += ((x[j - 1] / 2 + 1 / 4) - 0.5) ** 2
            g.append(sum_val)
        theta = np.array([(math.pi / 2 * (x[i] / 2 + 1 / 4)) for i in range(M - 1)])

        f = []
        for i in range(M):
            m = i + 1
            if m == 1:
                prod = np.prod([math.cos(val) for val in theta])
                f.append((prod) * (1 + g[i]))
            elif m < M:
                prod = np.prod([math.cos(val) for val in theta[:M - m]]) * (math.sin(theta[M - m]))
                f.append((prod) * (1 + g[i]))
            else:
                f.append(math.sin(theta[0]) * (1 + g[i]))
        out["F"] = f


class MaF3(ElementwiseProblem):

    def __init__(self, n_var=7, n_obj=3):
        xl = 0 * anp.ones(n_var)
        xu = 1 * anp.ones(n_var)

        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=0, xl=xl, xu=xu, elementwise_evaluation=True)

        # store custom variables needed for evaluation
        self.n_obj = n_obj
        self.n_var = n_var

    def _evaluate(self, x, out, *args, **kwargs):
        M = self.n_obj
        D = self.n_var
        g = D - M + 1
        for i in range(M - 1, D):
            g += (x[i] - 0.5) ** 2 - math.cos(20 * math.pi * (x[i] - 0.5))
        g = 100 * g
        f = []
        for i in range(M):
            m = i + 1
            if m == 1:
                prod = np.prod([math.cos(math.pi / 2 * x[j]) for j in range(M - m)])
                f.append(((prod) * (1 + g)) ** 4)
            elif m < M:
                prod = np.prod([math.cos(math.pi / 2 * x[j]) for j in range(M - m)]) * (
                    math.sin(math.pi / 2 * x[M - m]))
                f.append(((prod) * (1 + g)) ** 4)
            else:
                f.append((math.sin(math.pi / 2 * x[0]) * (1 + g)) ** 2)
        out["F"] = f


class MaF4(ElementwiseProblem):

    def __init__(self, n_var=7, n_obj=3):
        xl = 0 * anp.ones(n_var)
        xu = 1 * anp.ones(n_var)

        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=0, xl=xl, xu=xu, elementwise_evaluation=True)

        # store custom variables needed for evaluation
        self.n_obj = n_obj
        self.n_var = n_var

    def _evaluate(self, x, out, *args, **kwargs):
        M = self.n_obj
        D = self.n_var
        a = 2
        g = D - M + 1
        for i in range(M - 1, D):
            g += (x[i] - 0.5) ** 2 - math.cos(20 * math.pi * (x[i] - 0.5))
        g = 100 * g
        f = []
        for i in range(M):
            m = i + 1
            if m == 1:
                prod = np.prod([math.cos(math.pi / 2 * x[j]) for j in range(M - m)])
                f.append((a ** m) * (1 - prod) * (1 + g))
            elif m < M:
                prod = np.prod([math.cos(math.pi / 2 * x[j]) for j in range(M - m)]) * (
                    math.sin(math.pi / 2 * x[M - m]))
                f.append((a ** m) * (1 - prod) * (1 + g))
            else:
                f.append((a ** m) * (1 - math.sin(math.pi / 2 * x[0])) * (1 + g))
        out["F"] = f


class MaF5(ElementwiseProblem):

    def __init__(self, n_var=7, n_obj=3):
        xl = 0 * anp.ones(n_var)
        xu = 1 * anp.ones(n_var)

        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=0, xl=xl, xu=xu, elementwise_evaluation=True)

        # store custom variables needed for evaluation
        self.n_obj = n_obj
        self.n_var = n_var

    def _evaluate(self, x, out, *args, **kwargs):
        M = self.n_obj
        D = self.n_var
        a = 2
        alpha = 100
        g = 0
        for i in range(M - 1, D):
            g += (x[i] - 0.5) ** 2
        f = []
        for i in range(M):
            m = i + 1
            power = M - i
            if m == 1:
                prod = np.prod([math.cos(math.pi / 2 * (x[j] ** alpha)) for j in range(M - m)])
                f.append((a ** power) * ((prod) * (1 + g)) ** 4)
            elif m < M:
                prod = np.prod([math.cos(math.pi / 2 * (x[j] ** alpha)) for j in range(M - m)]) * (
                    math.sin(math.pi / 2 * (x[M - m] ** alpha)))
                f.append((a ** power) * ((prod) * (1 + g)) ** 4)
            else:
                f.append((a ** power) * ((math.sin(math.pi / 2 * (x[0] ** alpha))) * (1 + g)) ** 4)
        out["F"] = f


class MaF7(ElementwiseProblem):

    def __init__(self, n_var=7, n_obj=3):
        xl = 0 * anp.ones(n_var)
        xu = 1 * anp.ones(n_var)

        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=0, xl=xl, xu=xu, evaluation_of="auto")

        # store custom variables needed for evaluation
        self.n_obj = n_obj
        self.n_var = n_var

    def _evaluate(self, x, out, *args, **kwargs):
        M = self.n_obj
        D = self.n_var
        prob = get_problem('dtlz7', n_obj=M, n_var=D)
        f = prob.evaluate(x)
        out["F"] = f


class MaF8(ElementwiseProblem):

    def __init__(self, n_var=2, n_obj=3):
        n_var = 2  # mandatory
        xl = -10000 * anp.ones(n_var)
        xu = 10000 * anp.ones(n_var)

        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=0, xl=xl, xu=xu, elementwise_evaluation=True)

        # store custom variables needed for evaluation
        self.n_obj = n_obj
        self.n_var = n_var

    def _evaluate(self, x, out, *args, **kwargs):
        M = self.n_obj
        D = self.n_var
        theta = 2 * math.pi / M
        pt = []
        pt.append([0, 1])
        # m = pt[0]
        t = [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]
        for i in range(M - 1):
            x1 = np.matmul(pt[-1], t)
            pt.append(x1)
        pt = np.array(pt)

        # f=[]
        # for i in range(M):
        #     dist = euclidean(x, pt[i])
        #     f.append(dist)
        f = cdist([x], pt)[0]
        # print(f,x)
        out["F"] = f


class MaF9(ElementwiseProblem):

    def __init__(self, n_var=2, n_obj=3):
        n_var = 2  # mandatory
        xl = -10000 * anp.ones(n_var)
        xu = 10000 * anp.ones(n_var)

        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=0, xl=xl, xu=xu, elementwise_evaluation=True)

        # store custom variables needed for evaluation
        self.n_obj = n_obj
        self.n_var = n_var

    def _evaluate(self, x, out, *args, **kwargs):
        M = self.n_obj
        D = self.n_var
        theta = 2 * math.pi / M
        pt = []
        pt.append([0, 1])
        # m = pt[0]
        t = [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]
        for i in range(M - 1):
            x1 = np.matmul(pt[-1], t)
            pt.append(x1)
        pt = np.array(pt)

        f = []
        edge = euclidean(pt[1], pt[0])
        for i in range(M):
            if i < M - 1:
                pts = [pt[i], pt[i + 1]]
            else:
                pts = [pt[i], pt[0]]
            i_vector = pts[1] - pts[0]
            d_vector = x - pts[0]
            cos_component = np.dot(d_vector, i_vector) / edge
            dist = (np.linalg.norm(d_vector) ** 2 - cos_component ** 2) ** 0.5
            f.append(dist)
        # f = cdist([x],pt)[0]
        # print(f,x)
        out["F"] = f


class MaF13(ElementwiseProblem):

    def __init__(self, n_var=7, n_obj=3):
        xl = -2 * anp.ones(n_var)
        xu = 2 * anp.ones(n_var)
        xl[0] = 0
        xl[1] = 0
        xu[0] = 1
        xu[1] = 1

        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=0, xl=xl, xu=xu, elementwise_evaluation=True)

        # store custom variables needed for evaluation
        self.n_obj = n_obj
        self.n_var = n_var

    def _evaluate(self, x, out, *args, **kwargs):
        M = self.n_obj
        D = self.n_var

        j1 = [val for val in range(2, D) if (val + 1) % 3 == 1]
        j2 = [val for val in range(2, D) if (val + 1) % 3 == 2]
        j3 = [val for val in range(2, D) if (val + 1) % 3 == 0]
        j4 = [val for val in range(3, D)]

        y = []
        for i in range(D):
            temp = x[i] - 2 * x[1] * math.sin(2 * math.pi * x[0] + (i + 1) * math.pi / D)
            y.append(temp)
        y = np.array(y)

        f = []
        for i in range(M):
            m = i + 1
            if m == 1:
                temp = math.sin(math.pi / 2 * x[0]) + (2 / len(j1)) * (sum([y[j] ** 2 for j in j1]))
                f.append(temp)
            elif m == 2:
                temp = math.cos(math.pi / 2 * x[0]) * math.sin(math.pi / 2 * x[1]) + (2 / len(j2)) * (
                    sum([y[j] ** 2 for j in j2]))
                f.append(temp)
            elif m == 3:
                temp = math.cos(math.pi / 2 * x[0]) * math.cos(math.pi / 2 * x[1]) + (2 / len(j3)) * (
                    sum([y[j] ** 2 for j in j3]))
                f.append(temp)
            else:
                temp = f[0] ** 2 + f[1] ** 10 + f[2] ** 10 + (2 / len(j4)) * (sum([y[j] ** 2 for j in j4]))
                f.append(temp)
        out["F"] = f


class MaF14(ElementwiseProblem):

    def __init__(self, n_var=7, n_obj=3):
        n_var = 20 * n_obj
        xl = 0 * anp.ones(n_var)
        xu = 10 * anp.ones(n_var)
        for i in range(n_obj - 1):
            xu[i] = 1

        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=0, xl=xl, xu=xu, elementwise_evaluation=True)

        # store custom variables needed for evaluation
        self.n_obj = n_obj
        self.n_var = self.n_obj * 20

    def eta1(x_i):
        sum_val = 0
        for i in range(len(x_i)):
            sum_val += x_i[i] ** 2 - 10 * math.cos(2 * math.pi * x_i[i]) + 10
        return sum_val / len(x_i)

    def eta2(x_i):
        sum_val = 0
        for i in range(len(x_i) - 1):
            sum_val += 100 * (x_i[i] ** 2 - x_i[i + 1]) ** 2 + (x_i[i] - 1) ** 2
        return sum_val / len(x_i)

    def g_calc(m1, m2, Nk, size_group, size_subgroup, x, M):
        if m1 % 2 == 1:
            temp = 0
            for j in range(Nk):
                index = m2 * size_group + j * size_subgroup + M - 1
                x_i = x[index:index + size_subgroup]
                temp += MaF14.eta1(x_i)
            g = temp / Nk
        else:
            temp = 0
            for j in range(Nk):
                index = m2 * size_group + j * size_subgroup + M - 1
                x_i = x[index:index + size_subgroup]
                temp += MaF14.eta2(x_i)
            g = temp / Nk
        # print(g)
        return g

    def _evaluate(self, x, out, *args, **kwargs):
        M = self.n_obj
        D = self.n_var
        size_group = int(np.floor((D - M + 1) / M))
        Nk = 2
        size_subgroup = int(size_group / Nk)
        c = np.eye(M)

        f = []
        for i in range(M):
            m = i + 1
            if m == 1:
                sum_val = 1
                for ind in range(M):
                    sum_val += c[i, ind] * MaF14.g_calc(i + 1, ind, Nk, size_group, size_subgroup, x, M)
                prod = 1
                for ind in range(M - m):
                    prod = prod * x[ind]
                f.append(sum_val * prod)
            elif m < M:
                sum_val = 1
                for ind in range(M):
                    sum_val += c[i, ind] * MaF14.g_calc(i + 1, ind, Nk, size_group, size_subgroup, x, M)
                prod = (1 - x[M - m])
                for ind in range(M - m):
                    prod = prod * x[ind]
                f.append(sum_val * prod)
            else:
                sum_val = 1
                for ind in range(M):
                    sum_val += c[i, ind] * MaF14.g_calc(i + 1, ind, Nk, size_group, size_subgroup, x, M)
                prod = (1 - x[0])
                f.append(sum_val * prod)
        out["F"] = f


class MaF15(ElementwiseProblem):

    def __init__(self, n_var=7, n_obj=3):
        raise NotImplementedError
        n_var = 20 * n_obj
        xl = 0 * anp.ones(n_var)
        xu = 1 * anp.ones(n_var)

        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=0, xl=xl, xu=xu, elementwise_evaluation=True)

        # store custom variables needed for evaluation
        self.n_obj = n_obj
        self.n_var = self.n_obj * 20

    def eta1(x_i):
        sum_val = 0
        for i in range(len(x_i)):
            sum_val += x_i[i] ** 2 / 4000
        prod_val = 1
        for i in range(len(x_i)):
            prod_val = prod_val * (math.cos(x_i[i] / (i + 1) ** 0.5) + 1)
        sum_val -= prod_val
        return sum_val / len(x_i)

    def eta2(x_i):
        sum_val = 0
        for i in range(len(x_i) - 1):
            sum_val += x_i[i] ** 2
        return sum_val / len(x_i)

    def g_calc(m1, m2, Nk, size_group, size_subgroup, x, M):
        if m1 % 2 == 1:
            temp = 0
            for j in range(Nk):
                index = m2 * size_group + j * size_subgroup + M - 1
                x_i = x[index:index + size_subgroup]
                temp += MaF14.eta1(x_i)
            g = temp / Nk
        else:
            temp = 0
            for j in range(Nk):
                index = m2 * size_group + j * size_subgroup + M - 1
                x_i = x[index:index + size_subgroup]
                temp += MaF14.eta2(x_i)
            g = temp / Nk
        # print(g)
        return g

    def _evaluate(self, x, out, *args, **kwargs):
        M = self.n_obj
        D = self.n_var
        size_group = int(np.floor((D - M + 1) / M))
        Nk = 2
        size_subgroup = int(size_group / Nk)
        c = np.eye(M)
        for i in range(M):
            for j in range(M):
                if j == (i + 1):
                    c[i, j] = 1

        f = []
        for i in range(M):
            m = i + 1
            if m == 1:
                sum_val = 1
                for ind in range(M):
                    sum_val += c[i, ind] * MaF14.g_calc(i + 1, ind, Nk, size_group, size_subgroup, x, M)
                prod = 1
                for ind in range(M - m):
                    prod = prod * math.cos(x[ind] * math.pi / 2)
                f.append(sum_val * (1 - prod))
            elif m < M:
                sum_val = 1
                for ind in range(M):
                    sum_val += c[i, ind] * MaF14.g_calc(i + 1, ind, Nk, size_group, size_subgroup, x, M)
                prod = math.sin(x[ind] * math.pi / 2)
                for ind in range(M - m):
                    prod = prod * math.cos(x[ind] * math.pi / 2)
                f.append(sum_val * (1 - prod))
            else:
                sum_val = 1
                for ind in range(M):
                    sum_val += c[i, ind] * MaF14.g_calc(i + 1, ind, Nk, size_group, size_subgroup, x, M)
                prod = (math.sin(x[0] * math.pi / 2))
                f.append(sum_val * (1 - prod))
        out["F"] = f