from pathlib import Path

import numpy as np
from scipy.stats import multivariate_normal

from matplotlib import pyplot as plt

from plot.utils import config_mpl, get_save_path


def plot_gaussian_contour1():
    # Configure Matplotlib
    config_mpl()
    plt.rcParams["grid.linestyle"] = (0, (1, 5))

    # Data Preparing
    x = np.linspace(-3, 3, 60)
    y = np.linspace(-3, 3, 60)
    X, Y = np.meshgrid(x, y)

    pos = np.dstack((X, Y))
    mean = np.array([0, 0])
    diag = np.eye(2)
    fiag = np.flip(np.eye(2), axis=0)
    covs = [diag, diag + fiag * 0.5, diag + fiag * 0.8]
    rvs = [multivariate_normal(mean, cov * 0.54) for cov in covs]  # type: ignore
    Zs = [rv.pdf(pos) for rv in rvs]

    # Get Save Path
    save_path = get_save_path(Path(__file__))

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), dpi=500)
    _ = [ax.contour(X, Y, Z, cmap="jet", alpha=0.85) for ax, Z in zip(axes, Zs)]

    fig.savefig(save_path, format="svg", bbox_inches="tight", pad_inches=0.05)


if __name__ == "__main__":
    plot_gaussian_contour1()
