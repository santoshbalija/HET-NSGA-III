import numpy as np

from pyheterogeneous.core.util import all_targets, targets_to_pop
from pyheterogeneous.core.job import Job
from pymoo.constraints.tcv import TotalConstraintViolation
from pymoo.core.evaluator import Evaluator


def func_update_pop(job):
    pop = job["pop"]
    targets_to_pop(pop, job["out"])
    TotalConstraintViolation().do(pop, inplace=True)
    for ind in pop:
        for target in job.data["targets"]:
            ind.evaluated.add(target)


class HeterogeneousExpensiveEvaluator(Evaluator):

    def __init__(self,
                 scheduler,
                 func_callback=func_update_pop,
                 set_of_solutions="single",
                 target_values="single",
                 **kwargs):

        super().__init__(skip_already_evaluated=False, **kwargs)

        # the callback when the job is done
        self.func_callback = func_callback

        # defines if the set of solutions and the target values are evaluated in batch or elementwise
        self.set_of_solutions = set_of_solutions
        self.target_values = target_values

        # initialize the scheduler to be used for evaluations
        self.scheduler = scheduler

    def eval(self, problem, pop, targets=None, **kwargs):
        return super().eval(problem, pop, targets=targets, **kwargs)

    def _eval(self, problem, pop, evaluate_values_of, targets=None, wait_until_jobs_completed=None, **kwargs):
        if not self.scheduler.is_alive():
            return None

        if targets is None:
            targets = all_targets(problem.n_obj, problem.n_constr)

        elif isinstance(targets, tuple):
            targets = [targets]

        # initialize the population to make sure F and G and all others exist
        init_pop(pop, problem.n_obj, problem.n_constr)

        # the target groups for all evaluations
        target_groups = [target for target, _ in problem.eval_times]

        # create the jobs to be submitted
        jobs = create_jobs(problem, pop, targets, self.set_of_solutions, self.target_values, target_groups,
                           callback=self.func_callback)

        # submit all jobs to the scheduler and return the results
        for job in jobs:
            self.scheduler.submit(job)

        return jobs

    def __getstate__(self):
        state = self.__dict__.copy()
        state["scheduler"] = None
        return state


def init_pop(pop, n_obj, n_constr=0):
    # for each individual in pop make sure the structure exists
    for ind in pop:

        if len(ind.F) == 0:
            ind.F = np.full(n_obj, np.inf)

        if n_constr > 0:
            if len(ind.G) == 0:
                ind.G = np.full(n_constr, np.inf)

        if not ind.has("jobs") or ind.get("jobs") is None:
            ind.set("jobs", [])

    TotalConstraintViolation().do(pop, inplace=True)


def create_jobs(problem, pop, targets, set_of_solutions, target_values, target_groups, callback=None):
    jobs = []

    n, m = len(pop), len(targets)

    if target_values == "single":

        # which of the target groups to evaluate
        groups_to_eval = relevant_targets(target_groups, targets)

        if set_of_solutions == "single":
            for i in range(n):
                for j in groups_to_eval:
                    job = Job(problem=problem, pop=pop[[i]], targets=target_groups[j], callback=callback)
                    jobs.append(job)
        elif set_of_solutions == "batch":
            for j in groups_to_eval:
                job = Job(problem=problem, pop=pop, targets=target_groups[j], callback=callback)
                jobs.append(job)

    elif target_values == "batch":
        if set_of_solutions == "single":
            for i in range(n):
                job = Job(problem=problem, pop=pop[[i]], targets=targets, callback=callback)
                jobs.append(job)
        elif set_of_solutions == "batch":
            job = Job(problem=problem, pop=pop, targets=targets, callback=callback)
            jobs.append(job)

    return jobs


def relevant_targets(target_groups, targets):
    ret = []
    rem_targets = list(targets)
    while len(rem_targets) > 0:
        target = rem_targets.pop(0)

        for k, group in enumerate(target_groups):
            if target in group:
                break

        ret.append(k)
        rem_targets = [e for e in rem_targets if e not in group]

    return sorted(list(set(ret)))
