from pathlib import Path

import numpy as np
from scipy.stats import multivariate_normal

from matplotlib import pyplot as plt

from plot.utils import config_mpl, get_save_path


def get_trajectory(rv, init, tol=1e-7, max_iters=100):
    iters = [init]
    x, y = init
    z = rv.pdf([x, y])

    for _ in range(max_iters):
        # coordinate ascent
        # update y by argmax_z
        y_candidates = np.linspace(-3, 3, 1000)
        z_candidates = rv.pdf(np.array([[x, yc] for yc in y_candidates]))
        y = y_candidates[np.argmax(z_candidates)]
        new_z = rv.pdf([x, y])
        if abs(new_z - z) < tol:
            break
        z = new_z
        iters.append([x, y])
        # update x by argmax_z
        x_candidates = np.linspace(-3, 3, 1000)
        z_candidates = rv.pdf(np.array([[xc, y] for xc in x_candidates]))
        x = x_candidates[np.argmax(z_candidates)]
        new_z = rv.pdf([x, y])
        if abs(new_z - z) < tol:
            break
        z = new_z
        iters.append([x, y])

    return iters


def plot_coordinate_ascent_trajectory():
    # Configure Matplotlib
    config_mpl()
    plt.rcParams["grid.linestyle"] = (0, (1, 5))

    # Data Loading
    x = np.linspace(-3, 3, 60)
    y = np.linspace(-3, 3, 60)
    X, Y = np.meshgrid(x, y)

    pos = np.dstack((X, Y))
    mean = np.array([0, 0])
    diag = np.eye(2)
    fiag = np.flip(np.eye(2), axis=0)
    cov = diag + fiag * 0.5
    rv = multivariate_normal(mean, cov * 4)  # type: ignore
    Z = rv.pdf(pos)

    iters = get_trajectory(rv, init=[2, -2])
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

    plt.xlim(-2.4, 2.5)
    plt.xticks(np.arange(-2, 2.6, 0.5))
    plt.ylim(-2.4, 2.5)
    plt.yticks(np.arange(-2, 2.6, 0.5))
    plt.tick_params(axis="both", direction="in", top=True, right=True)

    plt.savefig(save_path, format="svg", bbox_inches="tight", pad_inches=0.05)


if __name__ == "__main__":
    plot_coordinate_ascent_trajectory()
