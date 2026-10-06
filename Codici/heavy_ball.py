import time
import numpy as np


def heavy_ball(nn, theta0, X, Y, alpha=1e-2, beta=0.9, eps=1e-8, t_max=5000):
    """
    Subgradient Heavy Ball (A1):
        p_t = -g_t + beta * p_{t-1},  theta_{t+1} = theta_t + alpha * p_t
    Stops when |f_{t+1} - f_t| / max(1, |f_t|) <= eps or after t_max iterations.

    Returns (theta, log) where log is a list of (f, ||g||, sparsity, time_per_iter).
    """
    theta = theta0.copy()
    p = np.zeros_like(theta)
    f, g = nn.oracle(theta, X, Y)
    log = []

    for t in range(t_max):
        t0 = time.perf_counter()
        p = -g + beta * p
        theta = theta + alpha * p
        f_new, g = nn.oracle(theta, X, Y)
        log.append((f_new, np.linalg.norm(g), int(np.sum(np.abs(theta) < 1e-3)), time.perf_counter() - t0)) # For every iteration it saves $f(\Theta_{t+1})$, norm_2(g_t+1), the sparsity, time needed 

        done = abs(f_new - f) / max(1.0, abs(f)) <= eps # Check
        f = f_new
        if done:
            break

    return theta, log