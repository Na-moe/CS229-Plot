from pathlib import Path

import numpy as np
from scipy.stats import multivariate_normal

from matplotlib import pyplot as plt

from plot.utils import config_mpl, get_save_path


def set_ax(ax):
    ax.set_xlim3d(-3, 3)
    ax.set_ylim3d(-3, 3)
    ax.set_zlim(0, 0.26)
    ax.xaxis.pane.set_alpha(0)
    ax.yaxis.pane.set_alpha(0)
    ax.zaxis.pane.set_alpha(0)
    ax.view_init(elev=30, azim=-115)
    ax.set_box_aspect([4, 4, 2])


def plot_gaussian_cov():
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
    rvs = [multivariate_normal(mean, cov) for cov in covs]  # type: ignore
    Zs = [rv.pdf(pos) for rv in rvs]

    # Get Save Path
    save_path = get_save_path(Path(__file__))

    # Plot
    fig = plt.figure(figsize=(20, 15), dpi=500)
    axes = [fig.add_subplot(int(f"13{i + 1}"), projection="3d") for i in range(3)]
    _ = [
        ax.plot_surface(
            X, Y, Z, cmap="jet", alpha=0.85, linewidth=0.3, edgecolors="black"
        )
        for ax, Z in zip(axes, Zs)
    ]
    _ = [set_ax(ax) for ax in axes]

    fig.savefig(save_path, format="svg", bbox_inches="tight", pad_inches=0.05)


if __name__ == "__main__":
    plot_gaussian_cov()
