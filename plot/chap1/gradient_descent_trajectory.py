from pathlib import Path

import numpy as np
from scipy.stats import multivariate_normal

from matplotlib import pyplot as plt

from plot.utils import config_mpl, get_save_path


def get_trajectory(rv, init, tol=1e-4, max_iters=100):
    iters = [init]
    x, y = init

    def fz(x, y):
        return rv.pdf([x, y]) * 1000

    for _ in range(max_iters):
        grad = np.array(
            [fz(x + tol, y) - fz(x - tol, y), fz(x, y + tol) - fz(x, y - tol)]
        ) / (2 * tol)
        x += grad[0] * 260
        y += grad[1] * 260
        if abs(x - 25) < 1e-6:
            break
        iters.append([x, y])

    return iters


def plot_gradient_descent_trajectory():
    # Configure Matplotlib
    config_mpl()
    plt.rcParams["grid.linestyle"] = (0, (1, 5))

    # Data Loading
    x = np.linspace(1, 50, 200)
    y = np.linspace(1, 50, 200)
    X, Y = np.meshgrid(x, y)

    pos = np.dstack((X, Y))
    mean = np.array([25, 25])
    diag = np.eye(2)
    fiag = np.flip(np.eye(2), axis=0)
    cov = (diag + fiag * 0.25) * 200
    rv = multivariate_normal(mean, cov)  # type: ignore #
    Z = rv.pdf(pos)

    iters = get_trajectory(rv, init=[48, 30])
    iters_array = np.array(iters)

    # Get Save Path
    save_path = get_save_path(Path(__file__))

    # Plot
    plt.figure(figsize=(8, 6.5), dpi=500)
    plt.contour(X, Y, Z, cmap="jet_r", alpha=0.85)
    plt.plot(
        iters_array[:, 0],
        iters_array[:, 1],
        "x-",
        color="blue",
        markersize=5,
        linewidth=1,
    )

    plt.xlim(1, 50)
    plt.xticks(np.arange(5, 51, 5))
    plt.ylim(1, 50)
    plt.yticks(np.arange(5, 51, 5))
    plt.tick_params(axis="both", direction="in", top=True, right=True)

    plt.savefig(save_path, format="svg", bbox_inches="tight", pad_inches=0.05)


if __name__ == "__main__":
    plot_gradient_descent_trajectory()
