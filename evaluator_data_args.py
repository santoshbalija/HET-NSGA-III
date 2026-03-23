from pymoo.factory import get_reference_directions
from pyheterogeneous.algorithms.he_rho_generalized_cons_first import HECF
from pyheterogeneous.core.display import HeterogeneousDisplay
from pyheterogeneous.core.problem import HeterogeneousExpensiveProblem
from pyheterogeneous.core.scheduler import SynchronousScheduler
from pyheterogeneous.core.termination import SchedulerTimeTermination
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.operators.sampling.lhs import LHS
import argparse
import os
import yaml
import joblib
from pymoo.core.callback import Callback
from pymoo.optimize import minimize
from time import perf_counter
from pymoo.problems.multi import *
from pymoo.problems.many import *
from AddProblems.additional_problems import *
from pymoo.problems.single import *

# Callback for saving generation data
callback_cols = ["f_pop", "time", "f_opt", "x", "n_evals", "n_gen", "archive_by_targets", 'evaluation_count_obj', 'evaluation_count_cons']
config_name = 'two_obj_two_const_cons_sigma'
conf = f"Data/{config_name}/"

class MyCallback(Callback):
    def __init__(self):
        super().__init__()
        self.count = 1
        self.data = {col: [] for col in callback_cols}
        self.cols = callback_cols

    def notify(self, algorithm):
        # Create the directory for saving data
        path = f"{conf}/{args.algorithm}/{args.problem}"
        os.makedirs(path, exist_ok=True)

        args_dict = vars(args)  # Convert the Namespace object to a dictionary
        config_dict = {
            'arguments': args_dict
        }

        # Save the config to a YAML file
        config_file = os.path.join(path, f'config_{args.algorithm}.yaml')
        with open(config_file, 'w') as file:
            yaml.dump(args_dict, file)
        # Append data to callback
        self.data["f_pop"].append(algorithm.pop.get("F"))
        if args.algorithm == 'base':
            # if sum of ET_f1...ET_fn=1 then time can be directly calculated
            # time= n_eval*(ET_f1+...+ET_fn)= n_eval
            self.data["time"].append(algorithm.evaluator.n_eval)
            self.data["f_opt"].append(algorithm.opt.get("F"))
            self.data["x"].append(algorithm.pop.get("X"))
            self.data["n_evals"].append(algorithm.evaluator.n_eval)
            self.data["n_gen"].append(algorithm.n_gen)
        else:
            self.data["time"].append(algorithm.scheduler.time())
            #self.data["selected_obj_indp"].append(algorithm.selected_solutions_obj)
            self.data["f_opt"].append(algorithm.opt.get("F"))
            self.data["x"].append(algorithm.pop.get("X"))
            self.data["n_evals"].append(algorithm.evaluator.n_eval)
            self.data["n_gen"].append(algorithm.n_gen)
            self.data["archive_by_targets"].append(algorithm.archive_by_targets)
            self.data["evaluation_count_obj"].append(algorithm.evaluation_count_obj)
            self.data["evaluation_count_cons"].append(algorithm.evaluation_count_cons)


if __name__ == "__main__":

    # arguments passer problem n_eval, n_obj
    parser = argparse.ArgumentParser(description='Test Problem param')
    parser.add_argument('--algorithm', type=str, default='HE_Cons_f', help='algorithm name')
    parser.add_argument('--problem', type=str, default='MW3(n_var=10)', help='Problem name')
    parser.add_argument('--n_obj', type=int, default=2, help='Number of objectives')
    parser.add_argument('--ET_f1', type=float, default=0.25, help='Evaluation time for objective 1')
    parser.add_argument('--ET_f2', type=float, default=0.25, help='Evaluation time for objective 2')
    parser.add_argument('--ET_g1', type=float, default=0.25, help='Evaluation time for constraint 1')
    parser.add_argument('--ET_g2', type=float, default=0.25, help='Evaluation time for constraint 2')
    parser.add_argument('--termination', type=int, default=300, help='Termination time')
    parser.add_argument('--surr_gen', type=int, default=30, help='Number of surrogate generations')
    parser.add_argument('--n_doe', type=int, default=100, help='Number of initial design points')
    # pop_size = n_infills = n_ref: all kept equal
    parser.add_argument('--pop_size', type=int, default=50, help='Surrogate population size')
    parser.add_argument('--n_infills', type=int, default=50, help='Number of infill points')
    parser.add_argument('--n_ref', type=int, default=50, help='Number of reference directions')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    # eta(sigma_factor) controls  the  influence of  surrogate uncertainty.
    parser.add_argument('--sigma_factor', type=float, default=20, help='Sigma_factor')
    # e(cons_sigma_level) must always be negative.
    parser.add_argument('--cons_sigma_level', type=float, default=-0.5, help='Constraint sigma level')



    # problem args
    args = parser.parse_args()
    algorithm = args.algorithm
    sigma_factor = args.sigma_factor
    cons_sigma_level = args.cons_sigma_level
    problem_name =eval(args.problem)
    n_obj = args.n_obj
    SEED = args.seed
    surr_gen = args.surr_gen
    n_doe = args.n_doe
    pop_size = args.pop_size
    n_infills = args.pop_size
    n_ref = args.n_ref

    et_f1 = args.ET_f1
    et_f2 = args.ET_f2
    et_g1 = args.ET_g1
    et_g2 = args.ET_g2
    eval_times = [[('F', 0), et_f1],
                  [('F', 1), et_f2],
                  [('G', 0), et_g1],
                  [('G', 1), et_g2]]
    problem = HeterogeneousExpensiveProblem(problem_name, eval_times)
    termination = SchedulerTimeTermination(args.termination)
    scheduler = SynchronousScheduler()

    defaults = dict(n_doe=n_doe, sampling=LHS(), surr_gen=surr_gen, pop_size=pop_size, n_infills=pop_size, n_offsprings=pop_size)
    kwargs = {**dict(ref_dirs=get_reference_directions("energy", n_obj, n_ref, seed=1)), **defaults}

    # print seed and problem and algorithm
    for arg, value in vars(args).items():
        print(f'{arg}: {value}')






    callback = MyCallback()
    # create the algorithm object
    path =f"{conf}/consig_{abs(args.cons_sigma_level)}/ET_{args.ET_f1}_{args.ET_f2}_{args.ET_g1}_{args.ET_g2}/{args.algorithm}/{args.problem}"
    os.makedirs(path, exist_ok=True)

    ##################### NSGA3 #####################################
    if algorithm == "base":
        algorithm_base = NSGA3(pop_size=kwargs.get('pop_size'),
                               sampling=LHS(),
                               ref_dirs=kwargs.get('ref_dirs'))

        # execute the optimization
        base = minimize(problem.problem,
                        algorithm_base,
                        seed=SEED,
                        termination=('n_eval', termination.max_time),
                        verbose=True,
                        callback=callback)

        try:

            joblib.dump(base.algorithm.callback.data, f"{path}/data_{args.seed}.pkl")

        except Exception as e:
            print(f"Error: {e}")

    #####################Surrogate based NSGA3 (SA-NSGA-III) #####################################
    if algorithm == "batch":
        algorithm_batch = HECF(heterogeneous=False,
                            surv_selection='surrogate',
                            use_surrogate=True,
                            display=HeterogeneousDisplay(),
                            scheduler=scheduler,
                            debug=False,
                            **kwargs)

        batch = minimize(problem,
                         algorithm_batch,
                         termination,
                         seed=SEED,
                         verbose=True,
                         callback=callback)
        try:

            joblib.dump(batch.algorithm.callback.data, f"{path}/data_{args.seed}.pkl")

        except Exception as e:
            print(f"Error: {e}")



    ##################################(MFE-NSGA-III/ HET-NSGA-III)#####################################################
    if algorithm == "HE_Cons_f":
        algorithm_pd_independent_et_dt_cf = HECF(heterogeneous=True,
                                             surv_selection='prob-independent-evaltime-alpha-time',
                                             use_surrogate=True,
                                             display=HeterogeneousDisplay(),
                                             scheduler=scheduler,
                                             sigma_factor=sigma_factor,
                                             cons_sigma_level=args.cons_sigma_level,
                                             debug=False,
                                             **kwargs)

        pd_independent_et_dt_cf = minimize(problem,
                                         algorithm_pd_independent_et_dt_cf,
                                         termination,
                                         seed=SEED,
                                         verbose=True,
                                         callback=callback)
        try:

            joblib.dump(pd_independent_et_dt_cf.algorithm.callback.data, f"{path}/data_{args.seed}.pkl")

        except Exception as e:
            print(f"Error: {e}")




