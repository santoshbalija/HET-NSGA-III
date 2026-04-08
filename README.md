# Draft version of HET-NSGA-III / MFE-NSGA-III

Code for two companion papers on surrogate-assisted evolutionary multi- and many-objective optimization with heterogeneous evaluation times:

**1 (HET-NSGA-III):** Balija Santoshkumar and Kalyanmoy Deb, "Handling Objectives with Heterogeneous Evaluation Times in Surrogate-Assisted Evolutionary Multi-Objective Optimization," *IEEE Transactions on Evolutionary Computation*, 2025. [[Link](https://doi.org/10.1109/TEVC.2025.3560922)]

**2 (MFE-NSGA-III):** Balija Santoshkumar and Kalyanmoy Deb, "Handling of Objectives and Constraints with Heterogeneous Evaluation Times for Surrogate-Assisted Evolutionary Multi- and Many-Objective Optimization," *Swarm and Evolutionary Computation*, vol. 100, 2026. [[Link](https://doi.org/10.1016/j.swevo.2025.102260)]

---

## Overview

In most surrogate-assisted evolutionary multi-objective optimization (SA-EMO) algorithms, every infill solution is evaluated with high-fidelity models for all objectives and constraints. This is wasteful when evaluation times differ across objectives and constraints, and when a solution is largely infeasible or its surrogate prediction error is small.

This repository implements a **mixed-fidelity evaluation (MFE)** framework that decides, for each population member individually, which objectives and constraints need high-fidelity evaluation. The decision is based on four factors:

1. Whether the solution is largely feasible, largely infeasible, or close to a constraint boundary
2. The probability of the solution being non-dominated within its reference vector neighborhood
3. The relative high-fidelity evaluation time of each objective and constraint
4. The surrogate model uncertainty for each objective

**MFE-NSGA-III**  extends **HET-NSGA-III**  to constrained problems. When no constraints are present, MFE-NSGA-III degenerates to HET-NSGA-III.

---

## Repository Structure

```
HET-NSGA-III/
├── pyheterogeneous/
│   ├── algorithms/
│   │   ├── he_rho_generalized_cons_first.py       # HECF  — MFE-NSGA-III 
│   │   ├── he_rho_generalized_cons_first_block.py # HECFB — MFE-NSGA-III with block evaluation
│   │   ├── Util_func.py                           # Selection utilities 
│   │   └── probability_independent.py             # P_s^m and PG_s computation 
│   └── core/
│       ├── evaluator.py
│       ├── scheduler.py
│       ├── problem.py
│       ├── display.py
│       ├── termination.py
│       └── util.py
├── AddProblems/                   # Additional test problems (MaF, engineering problems)
├── pymoo/                         # pymoo dev 0.5.1 (bundled)
├── pydacefit/                     # Bundled dependency
├── ezmodel/                       # Bundled dependency
├── pysamoo/                       # Surrogate modeling framework (bundled)
├── evaluator_data_args.py         # Run script for HECF (MFE-NSGA-III)
├── evaluator_data_args_block.py   # Run script for HECFB (block evaluation)
└── requirements.txt
```

---

## Algorithm Description

### Mixed-Fidelity Selection (MFS) Metric — Eq. (2)

For every feasible solution `s` and each objective `m`, the MFS metric is:

$$\rho_s^m = \left(1 + \widehat{ET}^m\right)^\alpha \cdot P_s^m \cdot \left(1 + \left(\frac{\sigma_s^m}{|\Delta f_m|}\right)^{1/\eta}\right)$$

where `P_s^m` is the probability of solution `s` being non-dominated among its reference vector neighbors, `σ_s^m` is the surrogate uncertainty, `|Δf_m|` is the range of objective `m` in the population, `η = 20`, and `α` adapts over time per Eq. (4).

### Constraint Classification — Eq. (1)

Every solution `s` is classified using:

$$Z_s^j = \frac{-\mu_s^j}{\sigma_s^j}$$

where `μ_s^j` and `σ_s^j` are the surrogate prediction and uncertainty for constraint `j`. A solution is **Class I** (feasible/near-boundary) if `min_j Z_s^j ≥ e` (default `e = -0.5`). Otherwise it is **Class II** (largely infeasible).

### Survival Selection Procedure (Algorithm 1)

- **Step 2** (`|Class I| ≥ N`): Select `N` solutions from Class I using the MFS metric per reference vector. For each selected solution, evaluate constraints `j` with `e ≤ Z_s^j ≤ -e` (near boundary) and the objective `m` with the highest `ρ_s^m`.
- **Step 3** (`0 < |Class I| < N`): Use Step 2 for all Class I solutions, fill remaining slots with best Class II solutions by `PG_s` (no HF evaluation for infeasible solutions).
- **Step 4** (`|Class I| = 0`): Select the solution with the highest `PG_s` and evaluate its least-violated constraint (`argmax_j Z_s^j`).

### Overall Constraint Satisfaction Probability — Eq. (3)

$$PG_s = \prod_{j=1}^{J} P(g_j(s) \leq 0) = \prod_{j=1}^{J} \Phi(Z_s^j)$$

### Alpha Update — Eq. (4)

$$\alpha = \frac{(T - \tau) - (T_{\max} - T)}{T_{\max} - \tau}$$

where `T` is elapsed time, `τ = N_DoE` is the initial DoE budget, and `T_max` is the total budget. This shifts emphasis from cheap to expensive objectives over time.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/santoshbalija/HET-NSGA-III.git
```

If you want to create a new conda environment (recommended), use the following commands:

```bash
conda create -n hetemo python=3.11
conda activate hetemo
```

Then install the necessary libraries:

```bash
cd HET-NSGA-III
pip install -r requirements.txt
```

---

## Usage

### MFE-NSGA-III (constrained, non-block)

```bash
python evaluator_data_args.py \
    --algorithm HE_Cons_f \
    --problem "MW3(n_var=10)" \
    --n_obj 2 \
    --ET_f1 0.7 --ET_f2 0.1 \
    --ET_g1 0.15 --ET_g2 0.05 \
    --termination 300 \
    --n_doe 100 \
    --pop_size 50 \
    --surr_gen 30 \
    --seed 1 \
    --sigma_factor 20 \
    --cons_sigma_level -0.5
```

**Key arguments:**

| Argument | Description | default |
|---|---|---|
| `--algorithm` | `HE_Cons_f` (MFE-NSGA-III) or `base` (NSGA-III) or `batch` (SA-NSGA-III) | — |
| `--problem` | Any pymoo problem, e.g. `MW3(n_var=10)`, `C2DTLZ2(n_var=12, n_obj=3)` | — |
| `--ET_f1..ET_fM` | Relative evaluation time for each objective (must sum with constraints to 1.0) | varies |
| `--ET_g1..ET_gJ` | Relative evaluation time for each constraint | varies |
| `--termination` | Total time budget `T_max` | 300 (2-3 obj), 400 (5-8 obj) |
| `--n_doe` | Initial DoE size `N_DoE` | 100 (2-3 obj), 150 (5-8 obj) |
| `--pop_size` | Population size `N` (= n_infills = n_ref) | 50 |
| `--surr_gen` | Surrogate generations `t_S` per iteration | 30 |
| `--sigma_factor` | Surrogate uncertainty exponent `η` in Eq. (2) | 20 |
| `--cons_sigma_level` | Constraint boundary threshold `e` in Step 1 (must be negative) | -0.5 |
| `--seed` | Random seed | varies |

Results are saved to `Data/two_obj_two_const_cons_sigma/` as `.pkl` files per seed.

### MFE-NSGA-III with Block Evaluation (Section 4, Paper 2)

Block evaluation applies when multiple objectives and/or constraints are always evaluated together by the same third-party software call.

```bash
python evaluator_data_args_block.py \
    --algorithm HE_Cons_f_block \
    --problem "Carside()" \
    --n_obj 3 \
    --ET_f1 0.05 --ET_f2 0.95 \
    --ET_g1 0.95 --ET_g2 0.95 \
    --block_obj_cons "[[1, 2, 3]]" \
    --termination 300 \
    --pop_size 50 \
    --seed 1
```

The `--block_obj_cons` argument specifies which objectives and constraints are evaluated together. Indices follow the convention: objectives are `0..M-1`, constraints are `M..M+J-1`. For example `[[1, 3]]` means objective index 1 (f2) and constraint index 3 (g2, i.e. M+1=3 for M=2) are always evaluated together. Multiple blocks: `[[0, 2, 4], [1, 3]]`.

---

## Experimental Setup (from papers)

All results in both papers use:
- Population size `N = 50`, reference directions = 50 (Riesz energy method)
- `T_max = 300` for 2–3 objective problems, `T_max = 400` for 5–8 objective problems
- `N_DoE = 100` for 2–3 objective problems, `N_DoE = 150` for 5–8 objective problems
- `t_S = 30` surrogate generations per iteration
- `η = 20`, `e = -0.5`
- 15 independent runs per problem scenario, IGD+ as performance metric
- Kriging with six model configurations (constant/linear/quadratic × ARD/no-ARD), best selected by 3-fold cross-validation MAE

---
IGD+ is computed on the feasible non-dominated front of the 
population, normalized to using the min/max of the true (referernce) Pareto front
## Test Problems

The following problems from the papers are supported out of the box via pymoo and `AddProblems/`:

**Constrained multi-objective:** C2DTLZ2, TNK, MW3, MW7, MW14  
**Engineering design:** Welded beam (M=2, J=4), Carside impact (M=3, J=10), Disc brake (M=2, J=5)  
**Unconstrained (degenerate case):** ZDT1, ZDT2, ZDT3, DTLZ2  
**Single-objective constrained (degenerate case):** G1, G4, G10

---

## Citation

If you use this code, please cite both papers:

```bibtex
@article{santoshdebtevc2025,
  author={Santoshkumar, Balija and Deb, Kalyanmoy},
  journal={IEEE Transactions on Evolutionary Computation}, 
  title={Handling Objectives With Heterogeneous Evaluation Times in Surrogate-Assisted Evolutionary Multi-Objective Optimization}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={Optimization;Linear programming;Computational modeling;Measurement;Predictive models;Evolutionary computation;Probabilistic logic;Computational efficiency;Vectors;Uncertainty;Heterogeneous Evaluation Times;Heterogeneous Objectives;Non-uniform Latency;Surrogate-assisted Optimization;Multi-objective Optimization;Machine Learning;Reference Vector Guided Probabilistic Dominance},
  doi={10.1109/TEVC.2025.3560922}}


@article{santoshdeb2026,
  title={Handling of Objectives and Constraints with Heterogeneous Evaluation Times for Surrogate-Assisted Evolutionary Multi- and Many-Objective Optimization},
  author={Santoshkumar, Balija and Deb, Kalyanmoy},
  journal={Swarm and Evolutionary Computation},
  volume={100},
  pages={102260},
  year={2026},
  publisher={Elsevier},
  doi={10.1016/j.swevo.2025.102260}
}
```

---

## Contact

Balija Santoshkumar — balijasa@msu.edu and santhoshkumar.balija@gmail.com  

COIN Lab, Michigan State University — https://www.coin-lab.org/
