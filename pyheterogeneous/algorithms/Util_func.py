# Description: This file contains the utility functions used in the algorithms

import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.constraints.tcv import TotalConstraintViolation
from pymoo.util.termination.no_termination import NoTermination
from pymoo.core.population import Population
from pymoo.algorithms.moo.nsga3 import NSGA3

def select_solutions_with_rho_cons(Rho, required_solutions,  reference_directions, Zg, ZG, e):

    selected_solutions_full = np.empty((0, 4))  # To store the selected solutions
    # [individual index, probability, reference direction, objective index]
    selected_solutions_obj = []  # selected solutions for objective
    selected_solutions_cons = []  # selected solutions for constraints
    selected_solutions_pop_idx = []  #
    selected_solutions = []  # To store the selected solutions
    while len(set(selected_solutions_pop_idx)) < required_solutions:

        remaining_members = Rho[~np.all(Rho == selected_solutions_full[:, None, :], axis=-1).any(axis=0)]

        for ref_dir in range(len(reference_directions)):
            ref_dir_members = remaining_members[remaining_members[:, 2] == ref_dir]

            if len(ref_dir_members) == 0:
                continue
            selected_solution = ref_dir_members[np.argmax(ref_dir_members[:, 1])]
            # selected_solution all constraints array
            zg = Zg[int(selected_solution[0])]

            # Step 1: Check if the largest Z_g^G is less than threshold e
            if np.max(zg) < -e:
                #print("Largest Z_g^s is less than the threshold. Do not evaluate the objective or constraints.")
                continue
            else:
                selected_solution_cons = []
                for g in range(len(zg)):
                    if e <= zg[g] <= -e:
                        #print('selected_solution is near the constraint boundary')
                        selected_solution_cons.append(g)
            selected_solution_obj=[]
            if np.min(zg) > e:
                #print('selected_solution is greater than the threshold')
                selected_solution_obj = int(selected_solution[3])


            selected_solution_pop_idx = int(selected_solution[0])
            #selected_solution_obj = int(selected_solution[3])

            # if the selected solution already exists in the selected_solutions list
            if not np.any(np.all(selected_solutions_full == selected_solution, axis=1)):
                selected_solutions_full = np.vstack((selected_solutions_full, selected_solution))
                selected_solutions_pop_idx.append(selected_solution_pop_idx)
                selected_solutions_obj.append(selected_solution_obj)
                selected_solutions_cons.append(selected_solution_cons)
                selected_solutions.append([selected_solution_pop_idx, selected_solution_obj, selected_solution_cons])
                if len(set(selected_solutions_pop_idx)) == required_solutions:
                    break


    return selected_solutions, selected_solutions_pop_idx, selected_solutions_obj,selected_solutions_cons
def select_solutions_with_rho_cons_abs(Rho, required_solutions,  reference_directions, Zg, e):

    selected_solutions_full = np.empty((0, 4))  # To store the selected solutions
    # [individual index, probability, reference direction, objective index]
    selected_solutions_obj = []  # selected solutions for objective
    selected_solutions_cons = []  # selected solutions for constraints
    selected_solutions_pop_idx = []  #
    selected_solutions = [] # To store the selected solutions

    while len(set(selected_solutions_pop_idx)) < required_solutions:

        remaining_members = Rho[~np.all(Rho == selected_solutions_full[:, None, :], axis=-1).any(axis=0)]

        for ref_dir in range(len(reference_directions)):
            ref_dir_members = remaining_members[remaining_members[:, 2] == ref_dir]

            if len(ref_dir_members) == 0:
                continue
            selected_solution = ref_dir_members[np.argmax(ref_dir_members[:, 1])]
            # selected_solution all constraints array
            zg = Zg[int(selected_solution[0])]
            selected_solution_cons = []
            # Step 1: Check if solution is far from the constraint boundary
            if np.min(abs(zg)) > abs(e):
                #print("Largest Z_g^s is less than the threshold. Do not evaluate the objective or constraints.")
                pass
            else:
                # if the solution is near the constraint boundary within the threshold
                for g in range(len(zg)):
                    if e <= zg[g] <= -e:
                        #print('selected_solution is near the constraint boundary')
                        selected_solution_cons.append(g)
            selected_solution_obj=[]
            # step 3: if solution is highly feasible evaluate the objective
            if np.min(zg) > e:
                #print('selected_solution is greater than the threshold')
                selected_solution_obj = int(selected_solution[3])

            if selected_solution_obj == [] and selected_solution_cons == []:
                continue
            else:
                selected_solution_pop_idx = int(selected_solution[0])
                # if the selected solution already exists in the selected_solutions list
                if not np.any(np.all(selected_solutions_full == selected_solution, axis=1)):
                    selected_solutions_full = np.vstack((selected_solutions_full, selected_solution))
                    selected_solutions_pop_idx.append(selected_solution_pop_idx)
                    selected_solutions_obj.append(selected_solution_obj)
                    selected_solutions_cons.append(selected_solution_cons)
                    selected_solutions.append([selected_solution_pop_idx, selected_solution_obj, selected_solution_cons])
                    if len(set(selected_solutions_pop_idx)) == required_solutions:
                        break


    return selected_solutions, selected_solutions_pop_idx, selected_solutions_obj,selected_solutions_cons


def select_solutions_with_rho_cons_first_abs(Rho, PG_overall, required_solutions,  reference_directions, Zg, e):

    selected_solutions_full = np.empty((0, 4))  # To store the selected solutions
    # [individual index, probability, reference direction, objective index]
    selected_solutions_obj = []  # selected solutions for objective
    selected_solutions_cons = []  # selected solutions for constraints
    selected_solutions_pop_idx = []  #
    selected_solutions = [] # To store the selected solutions
    # Classify members into two classes based on zg >=e
    class1_members_idx = np.where(np.min(Zg, axis=1) >= e)[0]
    # Remove class 1 members from Rho to get class 2 members
    class2_members_idx = np.where(~np.isin(np.arange(len(Zg)), class1_members_idx))[0]



    # class1_Rho = Rho[np.isin(Rho[:, 0], class1_members[:, 0])]
    class1_Rho = Rho[np.isin(Rho[:, 0], class1_members_idx)]

    class2_PG_overall = PG_overall[np.isin(PG_overall[:, 0], class2_members_idx)]
    # Extract the last column from Rho added to the class2_PG_overall
    # just maintain the same shape as class2_PG_overall no impact on the values
    last_column = Rho[:, -1].reshape(-1, 1)
    class2_PG = np.hstack((PG_overall, last_column))


    # checks if class 1 members are not there in class 2 members
    if np.any(np.isin(class1_members_idx, class2_members_idx)):
        print('error: class1 members are present in class2 members')

    while len(set(selected_solutions_pop_idx)) < required_solutions:
        # if class 1 members are empty break the loop: no feasible solutions
        if len(class1_Rho) == 0:
            break

        remaining_members = class1_Rho[~np.all(class1_Rho == selected_solutions_full[:, None, :], axis=-1).any(axis=0)]

        for ref_dir in range(len(reference_directions)):
            ref_dir_members = remaining_members[remaining_members[:, 2] == ref_dir]

            if len(ref_dir_members) == 0:
                continue
            selected_solution = ref_dir_members[np.argmax(ref_dir_members[:, 1])]
            # selected_solution all constraints array
            zg = Zg[int(selected_solution[0])]
            selected_solution_cons = []
            # Step 1: Check if solution is far from the constraint boundary
            if np.min(abs(zg)) > abs(e):
                #print("Largest Z_g^s is less than the threshold. Do not evaluate the objective or constraints.")
                pass
            else:
                # if the solution is near the constraint boundary within the threshold
                for g in range(len(zg)):
                    if e <= zg[g] <= -e:
                        #print('selected_solution is near the constraint boundary')
                        selected_solution_cons.append(g)
            selected_solution_obj=[]
            # step 3: if solution is highly feasible evaluate the objective
            if np.min(zg) > e:
                #print('selected_solution is greater than the threshold')
                selected_solution_obj = int(selected_solution[3])

            if selected_solution_obj == [] and selected_solution_cons == []:
                continue
            else:
                selected_solution_pop_idx = int(selected_solution[0])
                # if the selected solution already exists in the selected_solutions list
                if not np.any(np.all(selected_solutions_full == selected_solution, axis=1)):
                    selected_solutions_full = np.vstack((selected_solutions_full, selected_solution))
                    selected_solutions_pop_idx.append(selected_solution_pop_idx)
                    selected_solutions_obj.append(selected_solution_obj)
                    selected_solutions_cons.append(selected_solution_cons)
                    selected_solutions.append([selected_solution_pop_idx, selected_solution_obj, selected_solution_cons])
                    if len(set(selected_solutions_pop_idx)) == required_solutions:
                        break

    # Class 2 solutions: infeasible solutions
    if len(set(selected_solutions_pop_idx)) < required_solutions:
        class2_PG = class2_PG[::-1] # not needed but no harm in shuffling
        while len(set(selected_solutions_pop_idx)) < required_solutions:
            remaining_members = class2_PG[~np.all(class2_PG == selected_solutions_full[:, None, :], axis=-1).any(axis=0)]
            selected_solution = remaining_members[np.argmax(remaining_members[:, 1])]
            selected_solution_obj = []
            # if all solutions are infeasible, select solution with largest ZG
            if len(class1_members_idx) == 0:
                selected_solution_cons_Zg = Zg[int(selected_solution[0])]
                cons = [int(np.argmax(selected_solution_cons_Zg))]
                selected_solution_cons = cons
            else:
                selected_solution_cons = []

            selected_solution_pop_idx = int(selected_solution[0])
            # if the selected solution already exists in the selected_solutions list
            if not np.any(np.all(selected_solutions_full == selected_solution, axis=1)):
                selected_solutions_full = np.vstack((selected_solutions_full, selected_solution))
                selected_solutions_pop_idx.append(selected_solution_pop_idx)
                selected_solutions_obj.append(selected_solution_obj)
                selected_solutions_cons.append(selected_solution_cons)
                selected_solutions.append([selected_solution_pop_idx, selected_solution_obj, selected_solution_cons])
                if len(set(selected_solutions_pop_idx)) == required_solutions:
                    break

        # return selected_solutions, selected_solutions_pop_idx, selected_solutions_obj, selected_solutions_cons


    return selected_solutions, selected_solutions_pop_idx, selected_solutions_obj,selected_solutions_cons
def select_solutions_with_rho_cons_first_abs_block(Rho, PG_overall, required_solutions,  reference_directions, Zg, e,block_obj_cons, n_obj):

    selected_solutions_full = np.empty((0, 4))  # To store the selected solutions
    # [individual index, probability, reference direction, objective index]
    selected_solutions_obj = []  # selected solutions for objective
    selected_solutions_cons = []  # selected solutions for constraints
    selected_solutions_pop_idx = []  #
    selected_solutions = [] # To store the selected solutions
    # Classify members into two classes based on zg >=e
    class1_members_idx = np.where(np.min(Zg, axis=1) >= e)[0]
    # Remove class 1 members from Rho to get class 2 members
    class2_members_idx = np.where(~np.isin(np.arange(len(Zg)), class1_members_idx))[0]

    class1_Rho = Rho[np.isin(Rho[:, 0], class1_members_idx)]

    class2_PG_overall = PG_overall[np.isin(PG_overall[:, 0], class2_members_idx)]
    # Extract the last column from Rho added to the class2_PG_overall
    # just maintain the same shape as class2_PG_overall no impact on the values
    last_column = Rho[:, -1].reshape(-1, 1)
    class2_PG = np.hstack((PG_overall, last_column))


    # checks if class 1 members are not there in class 2 members
    if np.any(np.isin(class1_members_idx, class2_members_idx)):
        print('error: class1 members are present in class2 members')

    while len(set(selected_solutions_pop_idx)) < required_solutions:
        # if class 1 members are empty break the loop: no feasible solutions
        if len(class1_Rho) == 0:
            break

        remaining_members = class1_Rho[~np.all(class1_Rho == selected_solutions_full[:, None, :], axis=-1).any(axis=0)]

        for ref_dir in range(len(reference_directions)):
            ref_dir_members = remaining_members[remaining_members[:, 2] == ref_dir]

            if len(ref_dir_members) == 0:
                continue
            selected_solution = ref_dir_members[np.argmax(ref_dir_members[:, 1])]
            # selected_solution all constraints array
            zg = Zg[int(selected_solution[0])]
            selected_solution_cons = []
            # Step 1: Check if solution is far from the constraint boundary
            if np.min(abs(zg)) > abs(e):
                #print("Largest Z_g^s is less than the threshold. Do not evaluate the objective or constraints.")
                pass
            else:
                # if the solution is near the constraint boundary within the threshold
                for g in range(len(zg)):
                    if e <= zg[g] <= -e:
                        #print('selected_solution is near the constraint boundary')
                        selected_solution_cons.append(g)
            selected_solution_obj=[]
            # step 3: if solution is highly feasible evaluate the objective
            if np.min(zg) > e:
                #print('selected_solution is greater than the threshold')
                selected_solution_obj = int(selected_solution[3])

            if selected_solution_obj == [] and selected_solution_cons == []:
                continue
            else:
                selected_solution_pop_idx = int(selected_solution[0])
                # if the selected solution already exists in the selected_solutions list
                if not np.any(np.all(selected_solutions_full == selected_solution, axis=1)):
                    selected_solutions_full = np.vstack((selected_solutions_full, selected_solution))
                    selected_solutions_pop_idx.append(selected_solution_pop_idx)
                    selected_solutions_obj.append(selected_solution_obj)
                    selected_solutions_cons.append(selected_solution_cons)
                    selected_solutions.append([selected_solution_pop_idx, selected_solution_obj, selected_solution_cons])
                # Todo: check for multiple blocks and more than three in block
                for block in block_obj_cons:
                    if selected_solution_cons:
                        selected_solution_cons_m = [x + n_obj for x in selected_solution_cons]
                    else:
                        selected_solution_cons_m = []

                    if (selected_solution_obj in block) or any(x in block for x in selected_solution_cons_m):
                        remain_obj_cons = block.copy()
                        if selected_solution_obj in remain_obj_cons:
                            remain_obj_cons.remove(selected_solution_obj)
                        for x in selected_solution_cons_m:
                            if x in remain_obj_cons:
                                remain_obj_cons.remove(x)

                        if any(x >= n_obj for x in remain_obj_cons):
                            remain_cons = [x - n_obj if x >= n_obj else x for x in remain_obj_cons]
                        else:
                            remain_cons = []


                        remain_obj = [x for x in remain_obj_cons if x < n_obj]


                        for i in range(len(remain_cons)):
                            selected_solution_cons.append(remain_cons[i])
                            selected_solutions_cons[-1] = selected_solution_cons
                            selected_solutions[-1] = ([selected_solution_pop_idx, selected_solution_obj, selected_solution_cons])


                        for i in range(len(remain_obj)):
                            if not np.any((selected_solutions_full[:, 0] == selected_solution[0]) &
                                            (selected_solutions_full[:, -1] == remain_obj[i])):
                                selected_solutions_obj.append(remain_obj[i])
                                selected_solutions_pop_idx.append(selected_solution_pop_idx)
                                cons =[]
                                selected_solutions_cons.append(cons)
                                selected_solutions.append([selected_solution_pop_idx, remain_obj[i], cons])
                                selected_solution_mod = remaining_members[np.where(
                                    (remaining_members[:, 0] == selected_solution_pop_idx) & (
                                                remaining_members[:, 3] == remain_obj[i]))]
                                selected_solutions_full = np.vstack((selected_solutions_full, selected_solution_mod))


                if len(set(selected_solutions_pop_idx)) == required_solutions:
                    break

    # Class 2 solutions: infeasible solutions
    if len(set(selected_solutions_pop_idx)) < required_solutions:
        class2_PG = class2_PG[::-1]
        while len(set(selected_solutions_pop_idx)) < required_solutions:
            remaining_members = class2_PG[~np.all(class2_PG == selected_solutions_full[:, None, :], axis=-1).any(axis=0)]
            selected_solution = remaining_members[np.argmax(remaining_members[:, 1])]
            selected_solution_obj = []
            # if all solutions are infeasible, select solution with smallest ZG
            if len(class1_members_idx) == 0:
                selected_solution_cons_Zg = Zg[int(selected_solution[0])]
                cons = [int(np.argmax(selected_solution_cons_Zg))]
                selected_solution_cons = cons
            else:
                selected_solution_cons = []

            selected_solution_pop_idx = int(selected_solution[0])
            # if the selected solution already exists in the selected_solutions list
            if not np.any(np.all(selected_solutions_full == selected_solution, axis=1)):
                selected_solutions_full = np.vstack((selected_solutions_full, selected_solution))
                selected_solutions_pop_idx.append(selected_solution_pop_idx)
                selected_solutions_obj.append(selected_solution_obj)
                selected_solutions_cons.append(selected_solution_cons)
                selected_solutions.append([selected_solution_pop_idx, selected_solution_obj, selected_solution_cons])

                # Todo: check for multiple blocks and more than three in block
                for block in block_obj_cons:
                    if selected_solution_cons:
                        selected_solution_cons_m = [x + n_obj for x in selected_solution_cons]
                    else:
                        selected_solution_cons_m = []

                    if (selected_solution_obj in block) or any(x in block for x in selected_solution_cons_m):
                        remain_obj_cons = block.copy()
                        if selected_solution_obj in remain_obj_cons:
                            remain_obj_cons.remove(selected_solution_obj)
                        for x in selected_solution_cons_m:
                            if x in remain_obj_cons:
                                remain_obj_cons.remove(x)

                        if any(x >= n_obj for x in remain_obj_cons):
                            remain_cons = [x - n_obj if x >= n_obj else x for x in remain_obj_cons]
                        else:
                            remain_cons = []

                        remain_obj = [x for x in remain_obj_cons if x < n_obj]

                        for i in range(len(remain_cons)):
                            selected_solution_cons.append(remain_cons[i])
                            selected_solutions_cons[-1] = selected_solution_cons
                            selected_solutions[-1] = ([selected_solution_pop_idx, selected_solution_obj, selected_solution_cons])
                            selected_solution_mod = remaining_members[np.where(
                                (remaining_members[:, 0] == selected_solution_pop_idx) & (
                                        remaining_members[:, 3] == selected_solution_obj))]
                            selected_solutions_full = np.vstack((selected_solutions_full, selected_solution_mod))

                        for i in range(len(remain_obj)):
                            if not np.any((selected_solutions_full[:, 0] == selected_solution[0]) &
                                          (selected_solutions_full[:, -1] == remain_obj[i])):
                                selected_solutions_obj.append(remain_obj[i])
                                selected_solutions_pop_idx.append(selected_solution_pop_idx)
                                cons = []
                                selected_solutions_cons.append(cons)

                                selected_solutions.append([selected_solution_pop_idx, remain_obj[i], cons])
                                selected_solution_mod = remaining_members[np.where(
                                    (remaining_members[:, 0] == selected_solution_pop_idx) & (
                                            remaining_members[:, 3] == remain_obj[i]))]
                                selected_solutions_full = np.vstack((selected_solutions_full, selected_solution_mod))

                if len(set(selected_solutions_pop_idx)) == required_solutions:
                    break

        # return selected_solutions, selected_solutions_pop_idx, selected_solutions_obj, selected_solutions_cons


    return selected_solutions, selected_solutions_pop_idx, selected_solutions_obj,selected_solutions_cons

def select_solutions_with_rho_block(Rho, required_solutions,  reference_directions, block_obj):

    selected_solutions_full = np.empty((0, 4))  # To store the selected solutions
    selected_solutions_obj = []  # selected solutions
    selected_solutions_pop_idx = []  #
    selected_solutions = []  # To store the selected solutions
    while len(set(selected_solutions_pop_idx)) < required_solutions:

        remaining_members = Rho[~np.all(Rho == selected_solutions_full[:, None, :], axis=-1).any(axis=0)]

        for ref_dir in range(len(reference_directions)):
            ref_dir_members = remaining_members[remaining_members[:, 2] == ref_dir]

            if len(ref_dir_members) == 0:
                continue
            selected_solution = ref_dir_members[np.argmax(ref_dir_members[:, 1])]
            selected_solution_pop_idx = int(selected_solution[0])
            selected_solution_obj = int(selected_solution[3])

            # if selected_solution_obj in block_obj:
            #     remain_obj = block_obj
            #     remain_obj.remove(selected_solution_obj)
            #     selected_solution_obj.append(remain_obj)


            # if the selected solution already exists in the selected_solutions list
            if not np.any(np.all(selected_solutions_full == selected_solution, axis=1)):
                selected_solutions_full = np.vstack((selected_solutions_full, selected_solution))
                selected_solutions_pop_idx.append(selected_solution_pop_idx)
                selected_solutions_obj.append(selected_solution_obj)
                selected_solutions.append([selected_solution_pop_idx, selected_solution_obj])

                # Todo: check for multiple blocks and more than three in block
                for block in block_obj:
#                    block_obj = block

                    if selected_solution_obj in block:
                        remain_obj = block.copy()
                        remain_obj.remove(selected_solution_obj)
                        for i in range(len(remain_obj)):
                            selected_solutions_obj.append(remain_obj[i])
                            selected_solutions_pop_idx.append(selected_solution_pop_idx)
                            selected_solutions.append([selected_solution_pop_idx, remain_obj[i]])
                            selected_solution_mod =remaining_members[np.where((remaining_members[:, 0] == selected_solution_pop_idx) & (remaining_members[:, 3] == remain_obj[i]))]
                            selected_solutions_full = np.vstack((selected_solutions_full, selected_solution_mod))

                if len(set(selected_solutions_pop_idx)) == required_solutions:
                    break


    return selected_solutions, selected_solutions_pop_idx, selected_solutions_obj


def association_normalization(self, infills):
    """
    This function is used to normalize the infill solutions based on the hyperplane based boundary estimation
    Args:
        self:
        infills:

    Returns:

    """
    F = infills.get("F")
    # update the hyperplane based boundary estimation
    non_dominated = NonDominatedSorting().do(F, only_non_dominated_front=True)
    algorithm = NSGA3(ref_dirs=self.kwargs.get("ref_dirs"))
    ref_dirs = self.kwargs.get("ref_dirs")
    hyp_norm = algorithm.survival.norm
    hyp_norm.update(F, nds=non_dominated)
    ideal, nadir = hyp_norm.ideal_point, hyp_norm.nadir_point

    # associate the combined solutions to the reference directions using d2 distance
    niche_of_individuals, dist_to_niche, pbi, dist_matrix_d2, dist_matrix_pbi = \
        associate_to_niches(F, ref_dirs, ideal, nadir, theta=5, utopian_epsilon=0.0)
    return infills, niche_of_individuals, ref_dirs, dist_to_niche, pbi, dist_matrix_d2, dist_matrix_pbi, ideal, nadir


def surrogate_generations(self):
    # copy the current state of the algorithm and continue the optimization for a few iterations based on
    # surrogate model
    # it never uses the actual objective function evaluations
    """
    Args:
        self:

    Returns:

    """
    problem = self.surrogate.problem()
    # problem = self.problem.problem
    init_pop = copy_pop(self.pop)


    # copy the current state of the algorithm and continue the optimization for a few iterations
    # based on surrogate model
    algorithm = NSGA3(pop_size=self.kwargs.get("pop_size"),
                      sampling=init_pop,
                      ref_dirs=self.kwargs.get("ref_dirs"),
                     )
    algorithm.setup(problem, termination=NoTermination(),verbose=False)

    for _ in range(self.surr_gen):
        algorithm.next()
    pred_pop = algorithm.pop
    #print("infills-after-surr", pred_pop.get('F'))
    return pred_pop, init_pop



def associate_to_niches(F, niches, ideal_point, nadir_point,theta, utopian_epsilon=0.0):
    utopian_point = ideal_point - utopian_epsilon

    denom = nadir_point - utopian_point
    denom[denom == 0] = 1e-12

    # normalize by ideal point and intercepts
    N = (F - utopian_point) / denom
    dist_matrix_d1, dist_matrix_d2 = calc_distance(N, niches)
    niche_of_individuals = np.argmin(dist_matrix_d2, axis=1)

    dist_to_niche = dist_matrix_d2[np.arange(F.shape[0]), niche_of_individuals]
    dist_matrix_pbi = dist_matrix_d1+theta*dist_matrix_d2
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

def copy_pop(pop, attrs=None):
    if attrs is None:
        attrs = ["X", "F", "G",'evaluated']

    # cpy = Population.copy(pop, deep=True)
    cpy = Population(len(pop))

    for attr in attrs:
        vals = pop.get(attr)
        cpy.set(attr, np.copy(vals))

    TotalConstraintViolation().do(cpy, inplace=True)

    return cpy




