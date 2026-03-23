import time

import numpy as np

from pyheterogeneous.core.util import all_targets, get_target
from pymoo.problems.meta import MetaProblem


class HeterogeneousExpensiveProblem(MetaProblem):

    def __init__(self,
                 problem,
                 eval_times,
                 wait_until_time_passed=False
                 ):
        """

        This is a wrapper around test problems to benchmark different evaluation times.

        Parameters
        ----------
        problem : class
            The problem instance which shall be used to add time padding to target functions.

        eval_times : list
            The evaluation time of targets or target groups as a list.

        wait_until_time_passed : bool
            Whether the problem should simulate the true behaviour in each evaluation and wait until
            the evaluation would have been finished assuming the provided times for evaluation.

        """
        super().__init__(problem)

        # check the evaluation times - each target occurring exactly ones
        check_eval_times(self.n_obj, self.n_constr, eval_times)

        self.eval_times = eval_times
        self.wait_until_time_passed = wait_until_time_passed

    def evaluate(self, *args, **kwargs):
        kwargs["return_as_dictionary"] = True
        targets = kwargs.get("targets")

        # if no specific targets have been specific
        if targets is None:
            targets = all_targets(self.n_obj, self.n_constr)

        # if only a single target is supposed to be evaluated
        elif isinstance(targets, tuple):
            targets = [targets]

        kwargs["targets"] = targets
        kwargs["return_values_of"] = targets
        return super().evaluate(*args, **kwargs)

    def _evaluate(self, x, out, *args, targets=None, **kwargs):
        x = x[None, :] if x.ndim == 1 else x
        n, _ = x.shape

        # by default provide some output for F and G set to infinity
        out["F"] = np.full((len(x), self.n_obj), np.inf)
        out["G"] = np.full((len(x), self.n_constr), np.inf)

        # do the actual evaluation using the problem
        ret = self.problem.evaluate(x, *args, return_as_dictionary=True, **kwargs)

        # find all the groups to be evaluated
        I = set([find_target_group(self.eval_times, target) for target in targets])

        # now write each of the targets to the output
        for i in I:

            targets, eval_time = self.eval_times[i]

            for target in targets:
                out[target] = get_target(ret, target)

                _type, _k = target
                out[_type][:, _k] = out[target]

            if self.wait_until_time_passed:
                time.sleep(len(x) * eval_time)

        return out


def check_eval_times(n_obj, n_constr, eval_times):
    vals = []
    for k in range(len(eval_times)):

        targets, time = eval_times[k]

        if isinstance(targets, tuple):
            targets = [targets]
            eval_times[k] = [targets, time]

        vals.extend(targets)

    for m in range(n_obj):
        t = ("F", m)
        cnt = len([e for e in vals if e == t])
        assert cnt == 1, f"Each objective needs to occur exactly once in eval_times. But {t} does occur {cnt}!"

    for m in range(n_constr):
        t = ("G", m)
        cnt = len([e for e in vals if e == t])
        assert cnt == 1, f"Each constraint needs to occur exactly once in eval_times. But {t} does occur {cnt}!"

    return True


def find_target_group(eval_times, target):
    for k, (group, time) in enumerate(eval_times):
        if target in group:
            return k
