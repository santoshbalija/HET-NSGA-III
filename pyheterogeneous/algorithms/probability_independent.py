import numpy as np
import math

def compute_probability(Fr, Fc, Sr, Sc):
    """
    Compute the probability of improvement between two individuals for each objective.

    Args:
        Fr (numpy.ndarray): Objective function values of the reference individual.
        Fc (numpy.ndarray): Objective function values of the compared individual.
        Sr (numpy.ndarray): Uncertainties of the reference individual.
        Sc (numpy.ndarray): Uncertainties of the compared individual.

    Returns:
        list: List of probabilities of improvement for each objective.
    """
    probabilities = np.full((len(Fr), 1), np.nan)
    for l in range(len(Fr)):
        f = -(Fr[l] - Fc[l])
        s = Sr[l] ** 2 + Sc[l] ** 2
        prob = 0.5 * (1 + math.erf((-f) / (math.sqrt(2) * math.sqrt(s))))
        probabilities[l] = prob
    return probabilities

def compute_probability_cv(Gr,Sr):
    """
    Compute the probability of improvement between two individuals for each objective.

    Args:
        Gr (numpy.ndarray): Constraint function values of the reference individual.
        Sr (numpy.ndarray): Uncertainties of the reference individual.

    Returns:
        list: List of probabilities of improvement for each constraint.
    """
    probabilities = np.full((len(Gr)), np.nan)
    for l in range(len(Gr)):
        f = -(Gr[l])
        s = Sr[l]
        prob = 0.5 * (1 + math.erf((f) / (math.sqrt(2) * s)))
        probabilities[l] = prob

    return probabilities

def calculate_probabilities_independent(cand_combined, niche_of_individuals, ref_dirs):
    """
    Calculate the probabilities of improvement for individuals associated with each reference direction.

    Args:
        cand_combined (numpy.ndarray): Combined population.
        niche_of_individuals (numpy.ndarray): Niche assignment of individuals.
        ref_dirs (numpy.ndarray): Reference directions.

    Returns:
        list: List of probabilities [individual index, probability, reference direction].
    """

    Prob_all = []

    for i in range(len(ref_dirs)):
        # Find individuals associated with the current reference direction
        ref = np.where(niche_of_individuals == i)[0]

        if ref.shape[0] > 0:
            # Select the associated population
            pop = cand_combined[ref]
            Ft = pop.get("F")
            St = pop.get("F_sigma")

            if ref.shape[0] == 1:
                # If only one individual, set the probability as 0
                Prob_all.append([int(ref[0]), np.zeros((len(Ft[0]), 1)), i])
            else:
                Prob_r = []
                for j in range(len(ref)):
                    pop_idx = ref[j]
                    pop = cand_combined[pop_idx]
                    Fr = pop.get("F")
                    Sr = pop.get("F_sigma")

                    prob_m = np.zeros((len(Fr), 1))
                    for k in range(len(ref)):
                        if j != k:
                            pop_idx2 = ref[k]
                            pop2 = cand_combined[pop_idx2]
                            Fc = pop2.get("F")
                            Sc = pop2.get("F_sigma")
                            prob_f = compute_probability(Fr, Fc, Sr, Sc)
                            prob_m += prob_f
                    prob_t = prob_m / (len(ref) - 1)
                    Prob_r.append([int(pop_idx), prob_t, i])

                    Prob_all.append([int(pop_idx), prob_t, i])
    return Prob_all

def calculate_probabilities_independent_constr(cand_combined, niche_of_individuals, ref_dirs):
    """
    Calculate the probabilities of improvement for individuals associated with each reference direction.

    Args:
        cand_combined (numpy.ndarray): Combined population.
        niche_of_individuals (numpy.ndarray): Niche assignment of individuals.
        ref_dirs (numpy.ndarray): Reference directions.

    Returns:
        list: List of probabilities [individual index, probability, reference direction].
    """

    Prob_all_constr = []

    for i in range(len(ref_dirs)):
        # Find individuals associated with the current reference direction
        ref_associated = np.where(niche_of_individuals == i)[0]

        if ref_associated.shape[0] > 0:
            # Select the associated population
            pop = cand_combined[ref_associated]
            Prob_r_constr = []
            for j in range(len(ref_associated)):
                pop_idx = ref_associated[j]
                pop = cand_combined[pop_idx]
                Gr = pop.get("G")
                Sr = pop.get("G_sigma")

                prob_f_i = compute_probability_cv(Gr, Sr)
                # probability combined
                prob_f = np.prod(prob_f_i)

                Prob_r_constr.append([int(pop_idx), prob_f, i])

                Prob_all_constr.append([int(pop_idx), prob_f, i])
    return Prob_all_constr

