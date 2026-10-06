# Heavy Ball vs Proximal Bundle Method - Optimization Project

## Context & References
- Benchmark: Subgradient Heavy Ball (A1) vs Proximal Bundle Method (A2) on shallow neural network training with Lasso penalty.
- Reference report: `Project5_Pratesi_Jha.pdf` (notations, dimensions, analytical derivations).
- Model class: Existing neural network implementation in Python (already defines forward/backward and parameter vectorization).

## Mathematical Formulation & Invariants
- Objective: f(theta) = L(theta) + lambda * ||theta||_1
- Loss: Normalized MSE: L(theta) = (1/N) * sum_p ||y_p_hat - y_p||_2^2 with linear output units.
- Architecture: 1 hidden layer with tanh activation (sigma(t) = tanh(t)), dimension n = m*d + m + K*m + K.
- Subgradient selection: g_i(theta) = nabla L(theta)_i + lambda * sign(theta_i), with canonical choice sign(0) = 0.
- Hyperparameter: Structural lambda = 1e-3 (fixed, not tuned).

## Algorithmic Requirements
- Implementations must be from scratch (no high-level black-box ML/optimization library solvers for the main routines).
- A1 (Heavy Ball):
  - Momentum recurrence: p_t = -g_t + beta * p_{t-1}, theta_{t+1} = theta_t + alpha * p_t.
  - Stopping: Relative objective stagnation |f_{t+1} - f_t| / max(1, |f_t|) <= eps_HB or max iterations t_max.
- A2 (Proximal Bundle Method):
  - Master Problem: Solve the Wolfe Dual QP over the unit simplex Delta_{|B|} using off-the-shelf QP solver (e.g. cvxopt / scipy.optimize):
    min_{mu in Delta} (tau/2) * mu^T Q mu + alpha_tilde^T mu, where Q_ij = <g_i, g_j>.
  - Non-convexity fix: Enforce non-negative linearization errors via alpha_tilde_j = max(0, alpha_j).
  - Primal recovery: d* = -tau * sum_j mu_j* g_j.
  - Steps: Serious step if f(theta_bar) - f(theta+) >= m_L * (-v*), else Null step.
  - Adaptive tau: Increase tau after serious steps (gamma_inc), decrease after consecutive null steps (gamma_dec).
  - Stopping: Predicted descent -v* = tau ||g_agg||^2 + sum mu_j* alpha_tilde_j <= eps_bundle or max evaluations.

## Implementation & Coding Guidelines
- Write clean, concise, vectorized numerical code using standard scientific Python (NumPy).
- Maintain deterministic execution: enforce explicit random seeds across all experiments.
- Respect variable notation and shapes defined in the existing neural network `.py` script.
- Focus on numerical correctness and metrics logging (f(theta), ||g_agg||, sparsity count, execution time per iteration); no enterprise test suites or bloated scaffolding.