import copy
import numpy as np
from pyheterogeneous.core.display import HeterogeneousDisplay
from pyheterogeneous.core.evaluator import HeterogeneousExpensiveEvaluator
from pyheterogeneous.core.scheduler import SynchronousScheduler
from pyheterogeneous.core.util import pop_to_out, all_targets
from pymoo.constraints.tcv import TotalConstraintViolation
from pymoo.core.duplicate import DefaultDuplicateElimination
from pymoo.core.evaluator import Evaluator
from pymoo.core.population import Population
from pysamoo.core.surrogate import Surrogate
from pysamoo.core.target import Target
from pymoo.util.normalization import ZeroToOneNormalization
from pymoo.core.initialization import Initialization
from pymoo.operators.sampling.lhs import LHS
from pysamoo.core.defaults import DEFAULT_OBJ_MODELS, DEFAULT_IEQ_CONSTR_MODELS, DEFAULT_EQ_CONSTR_MODELS
from pyheterogeneous.algorithms.probability_independent import calculate_probabilities_independent, \
    calculate_probabilities_independent_constr
from pyheterogeneous.algorithms.Util_func import select_solutions_with_rho_cons_first_abs_block
from pyheterogeneous.algorithms.Util_func import association_normalization
from pyheterogeneous.algorithms.Util_func import surrogate_generations
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pyheterogeneous.core.util import targets_to_pop_index
from pymoo.core.algorithm import Algorithm
from pymoo.algorithms.moo.nsga3 import NSGA3
from time import perf_counter


class HECFB(Algorithm):

    def __init__(self,
                 *args,
                 n_doe=60,
                 n_infills=10,
                 surr_gen=30,
                 sampling=LHS(),
                 surv_selection="default",
                 use_surrogate=True,
                 order_targets_by="info_gain",
                 n_max_archive=1000000,
                 scheduler=SynchronousScheduler(),
                 display=HeterogeneousDisplay(),
                 heterogeneous=True,
                 debug=False,
                 selected_solutions_obj=None,
                 sigma_factor=20,
                 cons_sigma_level=-0.5,
                 evaluation_count_obj=None,
                 evaluation_count_cons=None,
                 block_obj_cons=[],
                 **kwargs):
        """

            This is the implementation of HE which make an algorithm being able to handle heterogenously expensive objectives.

            Parameters
            ----------
            n_doe : int
                Number of initial points to evaluate
            n_infills v: int
                The number of infill solutions in each generation
            gamma : int
                How often the probabilistic survival is repeated to calculate the survival probability.
            surr_gen : int
                In the probabilistic infill, how often it should be repeated.
            alpha_min : float
                The minimum survival probability of an individual to be kept during a target sequence evaluation.
            surv_selection : ["prob", "surrogate", "default"]
                The surv_selection type which is being used.
            use_surrogate : bool
                Whether a surrogate should be used at all (allows fallback to the algorithm with less overhead)
            order_targets_by : str
                What the targets should be ordered by.
            scheduler : class
                The scheduler responsible for the evaluation jobs
            display : class
                What to show in each iteration
            heterogeneous : bool
                Whether heterogeneous evaluation shall be used (default: True)
            sampling : class, pop, or X
                Set if the sampling should differ from the algorithm.

            """

        super().__init__(*args, display=display, **kwargs)
        self.args = args
        self.kwargs = kwargs
        # number of the initial design of experiments
        self.n_doe = n_doe

        # number of infills in each generation
        self.n_infills = n_infills

        # the initial sampling method
        self.sampling = sampling

        # whether a surrogate shall be used and updated
        self.use_surrogate = use_surrogate

        # how the ordering of targets should be determined
        self.order_targets_by = order_targets_by

        # the surv_selection being used to create offsprings
        self.surv_selection = surv_selection

        # the type of scheduler being used
        self.scheduler = scheduler

        # whether it should be exploited that the evaluation time varies
        self.heterogeneous = heterogeneous

        # a list of all targets
        self.V = None

        # an archive - here keeping track of each target separately
        self.archive = None

        # the prediction error for each target
        self.pred_error = None

        # the survival error for each target group
        self.surv_error = None

        # the prob
        self.pred_prob = None

        # the number of iterations on the surrogate
        self.surr_gen = surr_gen

        # whether some debug output shall be printed or not
        self.debug = debug

        # maximum length of the archive for each target
        self.n_max_archive = n_max_archive

        # an archive of solutions per target - all solutions evaluated on this specific target
        self.archive_by_targets = {}

        # a collections of points used for surrogate modeling for each target
        self.doe_by_targets = {}

        # the number of initial doe
        self.initialization = Initialization(sampling)

        # my callback
        # callback to be executed each generation
        self.he_callback = kwargs.get("he_callback")

        # the solution selected for high-fidelity evaluation
        self.selected_solutions_obj = selected_solutions_obj
        self.evaluation_count_obj = evaluation_count_obj
        self.evaluation_count_cons = evaluation_count_cons

        self.infills = None

        # the sigma multiplication factor for ET
        self.sigma_factor = sigma_factor
        self.time_initial = None
        self.fillup_time = None
        self.cons_sigma_level = cons_sigma_level

        # the block of objectives and constraints to be evaluated together
        self.block_obj_cons = block_obj_cons

    # -----------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------

    def _setup(self, problem, **kwargs):

        self.display = HeterogeneousDisplay()
        self.evaluator = HeterogeneousExpensiveEvaluator(self.scheduler,
                                                         set_of_solutions="single",
                                                         target_values="single",
                                                         )

        if self.use_surrogate:

            # the design space boundaries for the problem - used for normalization in the surrogate
            xl, xu = problem.bounds()
            defaults = dict(norm_X=ZeroToOneNormalization(xl, xu))

            targets = []

            models = DEFAULT_OBJ_MODELS(**defaults)
            for m in range(problem.n_obj):
                target = Target(("F", m), models)
                targets.append(target)

            models = DEFAULT_IEQ_CONSTR_MODELS(**defaults)
            for g in range(problem.n_ieq_constr):
                target = Target(("G", g), models)
                targets.append(target)

            models = DEFAULT_EQ_CONSTR_MODELS(**defaults)
            for h in range(problem.n_eq_constr):
                target = Target(("H", h), models)
                targets.append(target)

            # create the surrogate model
            self.surrogate = Surrogate(problem, targets)

        self.V = all_targets(self.problem.n_obj, self.problem.n_constr)

        self.archive_by_targets = {}
        for v in self.V:
            self.archive_by_targets[v] = Population()



        # get fill up budget to evaluate all objectives on pop size

        Tau = 0
        for t in range(len(self.problem.eval_times)):
            targets, ts = self.problem.eval_times[t]
            Tau += ts
        # together in same time
        t_block_rem_obj = 0
        for t_block in range(len(self.problem.eval_times)):
            for block in self.block_obj_cons:
                if t_block in block[1:]:
                    targets, ts = self.problem.eval_times[t_block]
                    t_block_rem_obj += ts
        # time required to evaluate all solutions on all targets

        self.fillup_time = (Tau-t_block_rem_obj) * self.n_infills

    def _initialize_infill(self):
        # use the algorithm to get the infills
        infills = self.initialization.do(self.problem, self.n_doe, algorithm=self)

        # if no doe are provided use the sampling method (default is LHS) to create the points
        if self.sampling is None:
            infills = self.sampling.do(self.problem, self.n_doe)

        return infills

    def _initialize_advance(self, infills=None, **kwargs):
        self.infills = infills

        # for block evaluation remove double time count since they will be evaluated in block their obj come
        # together in same time
        t_block_rem_obj = 0
        for t_block in range(len(self.problem.eval_times)):
            for block in self.block_obj_cons:
                if t_block in block[1:]:
                    targets, ts = self.problem.eval_times[t_block]
                    t_block_rem_obj += ts
        self.evaluator.scheduler.ts = self.evaluator.scheduler.ts - self.n_doe * t_block_rem_obj

        # update the archive for each target
        for v in self.V:
            self.archive_by_targets[v] = infills
            self.doe_by_targets[v] = infills

        # incase if you want reduce the initial population which is n_doe to n_infills
        if len(infills) > self.n_infills:
            algorithm = NSGA3(ref_dirs=self.kwargs.get("ref_dirs"))
            self.pop = algorithm.survival.do(self.problem, infills, n_survive=self.n_infills)

        # time spent on the initial design of experiments
        self.time_initial = self.evaluator.scheduler.ts

        # if a surrogate should be used
        if self.use_surrogate:
            # get the surrogate to be used for various purposes
            surrogate = self.surrogate

            # validate the different models for each target
            surrogate.validate(infills, indicator=["mae"])

            # also fit the best surrogate for each target
            surrogate.fit(infills)

            # get the prediction error (mae) for each target from the surrogate (initially from cross val)
            self.pred_error = {target.label: target.performance("mae") for target in surrogate.targets}

    def _infill(self):
        pass

    # -----------------------------------------------------------------------
    # CORE
    # -----------------------------------------------------------------------

    def _advance(self, **kwargs):

        # find the next solutions to evaluate on the problem or high-fidelity evaluations

        if self.surv_selection == "prob-independent-evaltime-alpha-time":
            infills_selected, infills_selected_idx, infills_selected_obj, infills_selected_cons, pred_infills = \
                self._prob_independent_evaltime_infill_alpha_time()

        elif self.surv_selection == "surrogate":
            pred_infills = self._surr_infill()
            for block in self.block_obj_cons:
                blocked_obj1 = block[0]
                targets, time_obj = self.problem.eval_times[blocked_obj1]
                self.evaluator.scheduler.ts = self.evaluator.scheduler.ts - (len(block) - 1) * time_obj * len(
                    pred_infills)
                print(self.evaluator.scheduler.ts)
            # block evaluation of objectives and constraints
        # whatever is done, store all solutions evaluated for the corresponding target
        sols_by_targets = {}
        self.evaluation_count_obj = []
        self.evaluation_count_cons = []

        # if the fact that independent evaluations are available should not be exploited
        if not self.heterogeneous:

            # if the method is surrogate batch we will not skip already evaluated solutions
            # we will evaluate all solutions on the problem
            if self.surv_selection == "default":
                infills = Population.new(X=pred_infills.get("X"))
                # take the infills and evaluate them directly on the problem
                self.evaluator.eval(self.problem, infills, target_values="batch", skip_already_evaluated=False)
                init_pop = copy_pop(self.pop)
                # add for each target all solutions to be evaluated
                for target in self.V:
                    sols_by_targets[target] = infills

            elif self.surv_selection == "surrogate":
                infills = Population.new(X=pred_infills.get("X"))
                # take the infills and evaluate them directly on the problem
                self.evaluator.eval(self.problem, infills, target_values="batch", skip_already_evaluated=False)
                print(self.evaluator.scheduler.ts)
                # add for each target all solutions to be evaluated
                for target in self.V:
                    sols_by_targets[target] = infills
            else:
                # take the infills and evaluate them directly on the problem
                # skip the ones which are already evaluated
                infills = copy_pop(pred_infills)
                self.evaluator.eval(self.problem, infills, target_values="batch", skip_already_evaluated=False)
                # add for each target all solutions to be evaluated
                for target in self.V:
                    sols_by_targets[target] = infills



        # otherwise perform a step-wise hetregenous evaluation

        else:

            # the number of objectives and constraints
            n_obj, n_constr = self.problem.n_obj, self.problem.n_constr

            sols_by_targets = {}

            for v in self.V:
                sols_by_targets[v] = empty_pop(pred_infills.get("X"), n_obj, n_constr)

            # the estimate error for each of the targets
            estm_error = dict(self.pred_error)

            # create empty population and offsprings populations which only contain true function values
            true_infills = empty_pop(pred_infills.get("X"), n_obj, n_constr)

            # get previous hi-fi evaluation status
            data = pred_infills.get("evaluated")

            # Modify the sets to keep only ('F', 0) , ('G', 0 )...and ('F', 1) set true evaluation status
            for i in range(len(data)):
                data[i] = {item for item in data[i] if isinstance(item, tuple) and item[0] in {'F', 'G'}}
            true_infills.set("evaluated", data)

            # pred_infills which are mixed with true function values as well
            infills_eval = copy_pop(pred_infills)

            if self.scheduler.time() <= (self.termination.max_time - self.fillup_time):

                print("doing step-wise evaluation")

                self.selected_solutions_obj = infills_selected_obj

                # now loop through the target evaluation
                for k in range(len(infills_selected)):

                    # if no more targets are left to be evaluated, break
                    if len(infills_eval) == 0:
                        # if nothing happened then are done

                        self.termination.force_termination = True

                        # nothing to do in this iteration
                        break

                    # objective function evaluation high-fidelity
                    if infills_selected_obj[k] != []:  # Check if the list is not empty
                    # the targets which are evaluated in the current order iteration
                        targets, time_obj = self.problem.eval_times[infills_selected_obj[k]]

                        # use the evaluator to obtain the exact values of the offsprings
                        self.evaluator.eval(self.problem, true_infills[infills_selected_idx[k]],
                                            targets=targets, target_values="single",
                                            evaluate_values_of=targets, skip_already_evaluated=True)



                    # if selected solution has been previously hi-fi evaluated take its value
                    # add it true infills
                        value = true_infills.get('F')[infills_selected_idx[k]][infills_selected_obj[k]]
                        if not np.isnan(value):
                            self.evaluation_count_obj.append(infills_selected_obj[k])
                            # block objectives are present just count time once remaining obj remove the times
                            # since they are evaluated together time is not counted
                            for block in self.block_obj_cons:
                                if infills_selected_obj[k] in block[1:]:
                                    self.evaluator.scheduler.ts = self.evaluator.scheduler.ts - time_obj
                            for target in targets:
                                sols_by_targets[target][infills_selected_idx[k]] = true_infills[
                                    infills_selected_idx[k]]

                                # from now on do not consider any estimation error for these targets
                                if target in estm_error:
                                    del estm_error[target]

                        if np.isnan(value):
                            out = pop_to_out(infills_eval, targets)
                            targets_to_pop_index(true_infills, out, infills_selected_idx[k])

                        # extract the values of high-fidelity evaluations
                        out = pop_to_out(true_infills, targets)
                        # copy the exact values from the high-fidelity population and mix with predictions
                        targets_to_pop_index(infills_eval, out, infills_selected_idx[k])


                    # constraint function evaluation high-fidelity

                    for cons in infills_selected_cons[k]:
                        # target  index
                        cons = cons + n_obj

                        if cons:  # Check if the list is not empty

                        # the targets which are evaluated in the current order iteration
                            targets, time_cons = self.problem.eval_times[cons]
                            t2_before = self.scheduler.time()

                            # use the evaluator to obtain the exact values of the offsprings
                            self.evaluator.eval(self.problem, true_infills[infills_selected_idx[k]],
                                                targets=targets, target_values="single",
                                                evaluate_values_of=targets, skip_already_evaluated=True)
                            t2_after = self.scheduler.time()



                        # if selected solution has been previously hi-fi evaluated take its value
                        # add it true infills
                            value = true_infills.get('G')[infills_selected_idx[k]][cons-n_obj]

                            if (t2_after-t2_before) > 0:
                                self.evaluation_count_cons.append(cons-n_obj)
                                # block objectives are present just count time once remaining obj remove the times
                                # since they are evaluated together time is not counted
                                for block in self.block_obj_cons:
                                    if cons in block[1:]:
                                        self.evaluator.scheduler.ts = self.evaluator.scheduler.ts - time_cons
                                for target in targets:
                                    sols_by_targets[target][infills_selected_idx[k]] = true_infills[
                                        infills_selected_idx[k]]

                                    # from now on do not consider any estimation error for these targets
                                    if target in estm_error:
                                        del estm_error[target]

                            if np.isnan(value):
                                out = pop_to_out(infills_eval, targets)
                                targets_to_pop_index(true_infills, out, infills_selected_idx[k])

                            # extract the values of high-fidelity evaluations
                            out = pop_to_out(true_infills, targets)
                            # copy the exact values from the high-fidelity population and mix with predictions
                            targets_to_pop_index(infills_eval, out, infills_selected_idx[k])



                # set evaluation status for high-fidelity evaluated
                infills_eval.set('evaluated', true_infills.get('evaluated'))

                infills_eval = infills_eval[list(set(infills_selected_idx))]
                pred_infills = pred_infills[list(set(infills_selected_idx))]


                # update sols by targets with the selected solutions
                for v in self.V:
                    sols_by_targets[v] = sols_by_targets[v][list(set(infills_selected_idx))]
                    # s=sols_by_targets[v].get('F')[:, v[1]]
                    # print(len(s[~np.isnan(s)]))
                obj_counts = [self.evaluation_count_obj.count(i) for i in range(n_obj)]
                cons_counts = [self.evaluation_count_cons.count(i) for i in range(n_constr)]
                for i in range(n_obj):
                    print(f"f{i + 1}-evalauted-count: {obj_counts[i]}")
                for i in range(n_constr):
                    print(f"g{i + 1}-evalauted-count: {cons_counts[i]}")


            # if the total time required to evaluate all solutions on all targets is less than the time left
            # last generation evaluate all solutions and skip already evaluated ones
            # this step is just for presentation of final solutions not part of actual algorithm
            else:
                print("time left", self.scheduler.time(), self.termination.max_time)
                print('evaluating all solutions')
                print('in benchmarking analysis take one generation less if time exceeds the budget')
                print('this step is just for presentation of final solutions')
                true_infills = true_infills[list(set(infills_selected_idx))]
                infills_eval = infills_eval[list(set(infills_selected_idx))]

                # evaluate all solutions on all targets
                for k in range(len(true_infills)):
                    for v in self.V:
                        #
                        # targets, time = self.problem.eval_times[v[1]]
                        # print(targets)
                        # print(v)
                        targets = [v]
                        # print("before evaluation", self.scheduler.time(), self.termination.max_time)
                        self.evaluator.eval(self.problem, true_infills[k], targets=targets, target_values="single",
                                            evaluate_values_of=targets, skip_already_evaluated=True)
                        # print("after evaluation", self.scheduler.time(), self.termination.max_time)
                        value = true_infills.get(v[0])[k][v[1]]
                         # process for keeping previously evaluated members objective function values
                        if not np.isnan(value):
                            if v[0] == 'F':
                                self.evaluation_count_obj.append(v[1])
                                _, time_obj = self.problem.eval_times[v[1]]
                                for block in self.block_obj_cons:
                                    if v[1] in block[1:]:
                                        self.evaluator.scheduler.ts = self.evaluator.scheduler.ts - time_obj
                            else:

                                self.evaluation_count_cons.append(v[1])
                                cons = v[1]+n_obj
                                _, time_cons = self.problem.eval_times[cons]
                                for block in self.block_obj_cons:
                                    if cons in block[1:]:
                                        self.evaluator.scheduler.ts = self.evaluator.scheduler.ts - time_cons

                        if np.isnan(value):
                            out = pop_to_out(infills_eval, targets)
                            targets_to_pop_index(true_infills, out, k)

                infills_eval = copy_pop(true_infills)
                infills_eval.set('evaluated', true_infills.get('evaluated'))
                self.pop = copy.deepcopy(infills_eval)
                nd_front = NonDominatedSorting().do(self.pop.get("F"), only_non_dominated_front=True)
                self.opt = self.pop[nd_front]
                self.termination.has_terminated = True
                # update the population with the true values and terminate and break the loop
                self.termination.force_termination = True


                # save all evaluated solutions into archive
                for target in self.V:
                    sols_by_targets[target] = infills_eval


            # update the infills with the true values submit them to the algorithm next generation
            infills = infills_eval

        print(self.scheduler.time(), self.termination.max_time)

        # update the values for each target

        # do the surrogate validation to find the best model for each target
        if self.use_surrogate:
            for target in self.surrogate.targets:
                v = target.label
                sols = sols_by_targets[v]
                if len(sols) > 0:
                    # Todo: check model selection vs computation time
                    comb = Population.merge(self.archive_by_targets[v], sols_by_targets[v])
                    t1_start = perf_counter()
                    # target.validate_modified(self.doe_by_targets[v], tst=sols, indicator=["mae"])

                    # splits the data into 5-fold cross-validation and selects best performing model
                    target.validate_modified(comb, find_best=True, indicator=["mae"])
                    self.pred_error[v] = target.performance("mae")
                    print(v, self.pred_error[v])
                    print(v, target.best)
                    t1_stop = perf_counter()
                    print("Elapsed time in validating:", t1_stop - t1_start)

                self.archive_by_targets[v] = Population.merge(self.archive_by_targets[v], sols_by_targets[v])[
                                             -self.n_max_archive:]


        # actually fit the surrogate for the next generation
        if self.use_surrogate:
            for target in self.surrogate.targets:
                v = target.label
                doe = self.archive_by_targets[v]
                # archive by target contains all the solutions evaluated on this target
                # it could nan values for the solutions not evaluated on this target
                # so the remove the nan values happens inside the surrogate.fit
                target.fit(doe)
                # print("after model selection")
                # print(f'X shape:{doe.get("X").shape}, F-shape{doe.get("F").shape}')
                # print("Fitted model", target.model)
                # print("Fitted model mae error", target.performance("mae"))

        # re-evaluate the solutions on the updated surrogate model
        # surrogate model is used to evaluate the solutions
        # problem as surrogate problem uses only surrogate model
        problem = self.surrogate.problem()
        Evaluator().eval(problem, infills, skip_already_evaluated=False)
        surr_infills = copy.deepcopy(infills)




        # update the population with the true values submit them to the algorithm next generation

        if len(infills) > 0:
            if self.termination.force_termination:
                # self.pop = infills
                pass
            else:

                self.off = copy.deepcopy(infills)
                # self.algorithm.advance(infills=infills)
                self.pop = copy.deepcopy(infills)

                # use the record to store the current plot

    def _set_optimum(self):
        nds = NonDominatedSorting().do(self.pop.get("F"), only_non_dominated_front=True)
        self.opt = self.pop[nds]

    # -----------------------------------------------------------------------
    # Utility Functions
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # Infills: Different kind of finding new solutions to be evaluated
    # -----------------------------------------------------------------------

    def _surr_infill(self):

        # copy the current state of the algorithm and continue the optimization for a few iterations based on
        # surrogate model
        # it never uses the actual objective function evaluations
        pred_pop, init_pop = surrogate_generations(self)

        return pred_pop

    def _prob_independent_evaltime_infill_alpha_time(self):

        # copy the current state of the algorithm and continue the optimization for a few iterations based on
        # surrogate model
        # it never uses the actual objective function evaluations
        pred_pop, init_pop = surrogate_generations(self)

        # eliminate the duplicates from the predicted population
        pred_pop_dup_e = DefaultDuplicateElimination().do(pred_pop, init_pop)

        # combine the initial population and the predicted population
        infills = Population.merge(init_pop, pred_pop_dup_e)

        # re-evaluating with  updated surrogate model no hifi eval
        # problem as surrogate problem uses only surrogate model
        problem = self.surrogate.problem()
        Evaluator().eval(problem, infills, skip_already_evaluated=False)

        # association of the combined solutions to the reference directions using d2 distance
        infills, niche_of_individuals, ref_dirs, dist_to_niche, pbi, dist_matrix_d2, dist_matrix_pbi, ideal, nadir \
            = association_normalization(self, infills)

        # calculate probability of dominance for each individual based reference directions
        Prob_r = calculate_probabilities_independent(infills, niche_of_individuals, ref_dirs)

        obj_wise = []
        for pop in Prob_r:

            for obj_id, prob in enumerate(pop[1].tolist()):
                modified_list = [pop[0]] + prob + [pop[2], obj_id]
                obj_wise.append(modified_list)

        Prob_stack = np.array(obj_wise)
        Prob_stack = Prob_stack[np.argsort(Prob_stack[:, -1], kind='mergesort')]


        Rho_et = Prob_stack.copy()

        # convert the probability of dominance to 1-P so that the probability of non-dominated can be calculated
        P_nd = 1 - Rho_et[:, 1]

        # overall constraint violation probability
        Prob_r_constr = calculate_probabilities_independent_constr(infills, niche_of_individuals, ref_dirs)

        Prob_r_constr = np.array(Prob_r_constr)


        Prob_r_constr_stack =np.vstack([Prob_r_constr for _ in range(problem.n_obj)])


        # Extract overall constraint violation probability
        PG_overall = Prob_r_constr_stack[:, 1]

        # constraint violation zscore
        # zg= -mu/sigma
        if problem.n_constr == 0:
            Zg = np.full((len(infills), 1), np.inf)

        else:
            Zg = (-infills.get("G")) / infills.get("G_sigma")
            # minimum value of Zg
            Zg_min = np.min(Zg)
            # maximum value of Zg
            Zg_max = np.max(Zg)

            # min(Z_g_s_min + 0.5 * (Z_g_s_max - Z_g_s_min), -1)
            # e = np.minimum(Zg_min + 0.5 * (Zg_max - Zg_min), -1)
        # e = np.minimum(np.median(Zg), self.cons_sigma_level)
        e = self.cons_sigma_level
        # print("Zg median",np.median(Zg))
        # print('Zg>e shape',Zg[np.min(Zg) > e].shape)


        # get sigma and make it 1d array f1 values are first and then f2 values
        Sigma = infills.get("F_sigma")

        # sort the sigma values in reference direction order so that we can multiply the sigma with the probability
        Sigma_sorted = Sigma[Rho_et[:len(infills), 0].astype(int)]


        normalization_factor = nadir - ideal
        Sigma_n = Sigma_sorted / normalization_factor

        # get sigma and make it 1d array f1 values are first and then f2 values

        Sigma_flat = Sigma_n.flatten('F')
        #print("Sigma_flat", Sigma_flat)

        # multiply the probability of dominance with the sigma
        Rho_Sigma = P_nd * (1 + Sigma_flat ** (1 / self.sigma_factor))


        # Finally save the row
        Rho_et[:, 1] = Rho_Sigma

        # first extract the evaluation times
        eval_times = self.problem.eval_times
        times = np.array([time for _, time in eval_times])

        # normalize the times
        times_n = (times / np.max(times))

        alpha = ((self.scheduler.ts - self.time_initial) - (self.termination.max_time - self.scheduler.ts)) / (
                    self.termination.max_time - self.time_initial)
        times = (1 + times_n) ** alpha
        for i in range(len(times)):
            start_idx = i * len(infills)
            end_idx = (i + 1) * len(infills)
            Rho_et[start_idx:end_idx, 1] = Rho_et[start_idx:end_idx, 1]*times[i]
        # select the required number of solutions going along the reference directions
        required_solutions = self.n_infills
        n_obj, n_constr = self.problem.n_obj, self.problem.n_constr
        # select members of each reference direction based on the probability of dominance and rho
        infills_selected, infills_selected_idx, infills_selected_obj, infills_selected_cons = select_solutions_with_rho_cons_first_abs_block(Rho_et,  Prob_r_constr_stack,
                                                                                                 required_solutions,
                                                                                                 ref_dirs, Zg,  e, block_obj_cons=self.block_obj_cons, n_obj=n_obj)

        return infills_selected, infills_selected_idx, infills_selected_obj, infills_selected_cons, infills


def noisy(sols, error):
    out = {}
    for type in ["F", "G", "H"]:
        out[type] = np.copy(sols.get(type))

    for (type, k), std in error.items():
        out[type][:, k] += np.random.normal(loc=0.0, scale=std, size=len(sols))

    noisy = Population.new(**out)
    TotalConstraintViolation().do(noisy, inplace=True)
    return noisy


def multiply_nested_lists(lst):
    result = []
    P_CV = []
    for sub_lst in lst:
        for sub_lst1 in sub_lst:
            # Multiply all elements in the nested array
            product = np.prod(sub_lst1[1])
            P_CV.append(sub_lst1[1])
            result.append([sub_lst1[0], product, sub_lst1[2]])
    return result, P_CV


def nested_lists(lst):
    P_CV_ind = []
    for sub_lst in lst:
        P_CV_ind.append(sub_lst[1])
    return P_CV_ind


def copy_pop(pop, attrs=None):
    if attrs is None:
        attrs = ["X", "F", "G", 'evaluated']

    # cpy = Population.copy(pop, deep=True)
    cpy = Population(len(pop))

    for attr in attrs:
        vals = pop.get(attr)
        cpy.set(attr, np.copy(vals))

    TotalConstraintViolation().do(cpy, inplace=True)

    return cpy


def empty_pop(X, n_obj, n_constr):
    n, _ = X.shape
    return Population.new(X=X, F=np.full((n, n_obj), np.nan), G=np.full((n, n_constr), np.nan))


def fill_nan(true_pop, pred_pop, copy=True):
    pop = copy_pop(true_pop) if copy else true_pop
    true_F, true_G = pop.get("F", "G")
    pred_F, pred_G = pred_pop.get("F", "G")

    fill = np.isnan(true_F)
    true_F[fill] = pred_F[fill]
    pop.set("F", true_F)

    if true_G.shape[1] > 0:
        fill = np.isnan(true_G)
        true_G[fill] = pred_G[fill]
        pop.set("G", true_G)

    TotalConstraintViolation().do(pop, inplace=True)

    return pop


def associate_to_niches(F, niches, ideal_point, nadir_point, theta, utopian_epsilon=0.0):
    utopian_point = ideal_point - utopian_epsilon

    denom = nadir_point - utopian_point
    denom[denom == 0] = 1e-12

    # normalize by ideal point and intercepts
    N = (F - utopian_point) / denom
    dist_matrix_d1, dist_matrix_d2 = calc_distance(N, niches)
    niche_of_individuals = np.argmin(dist_matrix_d2, axis=1)

    dist_to_niche = dist_matrix_d2[np.arange(F.shape[0]), niche_of_individuals]
    dist_matrix_pbi = dist_matrix_d1 + theta * dist_matrix_d2
    pbi = dist_matrix_pbi[np.arange(F.shape[0]), niche_of_individuals]

    return niche_of_individuals, dist_to_niche, pbi, dist_matrix_d2, dist_matrix_pbi


def calc_distance(N, ref_dirs):
    u = np.tile(ref_dirs, (len(N), 1))
    v = np.repeat(N, len(ref_dirs), axis=0)

    norm_u = np.linalg.norm(u, axis=1)

    scalar_proj = np.sum(v * u, axis=1) / norm_u
    proj = scalar_proj[:, None] * u / norm_u[:, None]
    val = np.linalg.norm(proj - v, axis=1)
    d1 = np.linalg.norm(proj, axis=1)
    d1 = np.reshape(d1, (len(N), len(ref_dirs)))
    d2 = np.reshape(val, (len(N), len(ref_dirs)))

    return d1, d2



