import time
import numpy as np
from scipy.optimize import minimize


def solve_master(G, alpha_t, tau):
    """Wolfe dual: min_{mu in simplex} (tau/2) mu^T Q mu + alpha_t^T mu, with Q = G G^T."""
    B = len(alpha_t)
    Q = G @ G.T
    res = minimize(lambda mu: 0.5 * tau * mu @ Q @ mu + alpha_t @ mu, # Function to minimize
                   np.full(B, 1.0 / B), # Creates an array of size B, each slot with value 1/B
                   jac=lambda mu: tau * Q @ mu + alpha_t, # Jacobian formula
                   bounds=[(0.0, 1.0)] * B, # each mu_i is bounded between 0 and 1
                   constraints={"type": "eq", "fun": lambda mu: mu.sum() - 1.0, "jac": lambda mu: np.ones(B)}, # Constraints imposed by the Wolf dual
                   method="SLSQP")
    mu = np.clip(res.x, 0.0, None)
    return mu / mu.sum()


def prox_bundle(nn, theta0, X, Y, tau=1.0, m_L=0.1, gamma_inc=1.5, gamma_dec=2.0, null_max=3,
                eps=1e-6, max_evals=2000, max_bundle=50):
    """
    Proximal Bundle Method (A2) with non-negative linearization errors alpha_tilde = max(0, alpha).
    Master: mu* = argmin_{mu in simplex} (tau/2) mu^T Q mu + alpha_tilde^T mu
            d* = -tau * g_agg,  g_agg = sum_j mu_j g_j,  -v* = tau ||g_agg||^2 + mu^T alpha_tilde
    Serious step if f(theta_bar) - f(theta+) >= m_L * (-v*), otherwise null step.
    Stops when -v* <= eps or after max_evals oracle calls.

    Returns (theta_bar, log) where log is a list of (f_bar, ||g_agg||, sparsity, time_per_iter).
    """
    theta = theta0.copy()
    f, g = nn.oracle(theta, X, Y)
    G = g[None, :]                # bundle subgradients, shape (|B|, n)
    alpha = np.zeros(1)           # linearization errors w.r.t. the current center theta_bar
    n_null = 0                    # Count of consecutive null steps
    log = []

    for evals in range(1, max_evals):
        t0 = time.perf_counter()
        alpha_t = np.maximum(0.0, alpha)
        mu = solve_master(G, alpha_t, tau)
        g_agg = mu @ G
        pred = tau * g_agg @ g_agg + mu @ alpha_t      # comutes the predicted descent -v*

        if pred <= eps:
            log.append((f, np.linalg.norm(g_agg), int(np.sum(np.abs(theta) < 1e-3)), time.perf_counter() - t0)) # Appends $f(\Theta_{t+1})$, norm_2(g_t+1), the sparsity, time needed
            break

        d = -tau * g_agg # Computes the vector of direction increase
        f_new, g_new = nn.oracle(theta + d, X, Y) # NN valuation on new point

        if f - f_new >= m_L * pred:                    # serious step: move the center
            alpha = alpha + (f_new - f) - G @ d        # re-center old linearization errors
            alpha_new = 0.0
            theta, f = theta + d, f_new
            tau *= gamma_inc
            n_null = 0
        else:                                          # null step: only enrich the model
            alpha_new = f - f_new + g_new @ d
            n_null += 1
            if n_null >= null_max:
                tau /= gamma_dec
                n_null = 0

        G = np.vstack([G, g_new])[-max_bundle:] # g_new gets added under G and we keep only the last 'max_bundle' rows
        alpha = np.append(alpha, alpha_new)[-max_bundle:] # alpha_new gets added at the end of alpha and we keep only the last 'max_bundle' rows
        log.append((f, np.linalg.norm(g_agg), int(np.sum(np.abs(theta) < 1e-3)), time.perf_counter() - t0)) # Appends current loss, norm_2(g_t+1), the sparsity, time needed

    return theta, log