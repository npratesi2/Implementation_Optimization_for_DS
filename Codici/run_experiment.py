import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from NN import NeuralNetwork
from heavy_ball import heavy_ball
from prox_bundle import prox_bundle

SEED, M, LAM = 0, 10, 1e-3

# Data: standardized features + string class label -> X (N, d), one-hot Y (N, K)
df = pd.read_csv("../Dataset/dry_bean_sample.csv")
X = df.drop(columns="Class").to_numpy()
classes = sorted(df["Class"].unique())
Y = np.eye(len(classes))[df["Class"].map({c: i for i, c in enumerate(classes)}).to_numpy()]

nn = NeuralNetwork(X.shape[1], M, len(classes), LAM)
theta0 = np.random.default_rng(SEED).normal(0.0, 0.1, nn.n_params)
f0, _ = nn.oracle(theta0, X, Y)
print(f"N={X.shape[0]}, d={X.shape[1]}, K={Y.shape[1]}, n={nn.n_params}, f(theta0)={f0:.4f}")

runs = {
    "Heavy Ball": heavy_ball(nn, theta0, X, Y, alpha=1e-2, beta=0.9, eps=1e-8, t_max=2000),
    "Prox Bundle": prox_bundle(nn, theta0, X, Y, eps=1e-6, max_evals=300),
}

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for name, (theta, log) in runs.items():
    f, gnorm, sparsity, times = map(np.array, zip(*log))
    print(f"{name:12s} iters={len(log):5d}  f={f[-1]:.4f}  ||g||={gnorm[-1]:.2e}  "
          f"zeros={sparsity[-1]}/{nn.n_params}  acc={nn.compute_accuracy(theta, X, Y):.1f}%  "
          f"time={times.sum():.2f}s ({1e3 * times.mean():.2f} ms/iter)")
    ax[0].plot(f, label=name)
    ax[1].semilogy(gnorm, label=name)
ax[0].set(xlabel="iteration", ylabel="f(theta)", title="Objective"); ax[0].legend()
ax[1].set(xlabel="iteration", ylabel="||g||", title="Subgradient norm")
plt.tight_layout(); plt.savefig("check.png", dpi=120)
