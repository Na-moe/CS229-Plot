from pathlib import Path

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import pandas as pd

from matplotlib import pyplot as plt

from plot.utils import config_mpl, get_save_path


def set_ax(ax):
    ax.set_xlim3d(-3, 3)
    ax.set_ylim3d(-3, 3)
    ax.set_zlim(0, 0.26)
    ax.xaxis.pane.set_alpha(0)
    ax.yaxis.pane.set_alpha(0)
    ax.zaxis.pane.set_alpha(0)
    ax.view_init(elev=15, azim=-125)


def plot_gda():
    # Configure Matplotlib
    config_mpl()
    plt.rcParams["grid.linestyle"] = (0, (1, 5))

    # Data Preparing
    data1 = pd.read_csv("data/chap4/gda1.csv", dtype=float)
    data2 = pd.read_csv("data/chap4/gda2.csv", dtype=float)

    # scatter points
    x1, y1 = data1["x"], data1["y"]
    x2, y2 = data2["x"], data2["y"]

    # decision boundary
    x = np.linspace(-1, 7, 100)
    y = -x

    # GDA
    X = np.vstack((data1, data2))
    Y = np.hstack((np.zeros(data1.shape[0]), np.ones(data2.shape[0])))
    lda = LDA(store_covariance=True)
    lda.fit(X, Y)
    mean1, mean2 = lda.means_  # type: ignore
    cov = lda.covariance_

    XX, YY = np.meshgrid(x, y)
    gs1 = multivariate_normal(mean1, cov)  # type: ignore
    Z1 = gs1.pdf(np.dstack((XX, YY)))

    gs2 = multivariate_normal(mean2, cov)  # type: ignore
    Z2 = gs2.pdf(np.dstack((XX, YY)))

    # Get Save Path
    save_path = get_save_path(Path(__file__))

    # Plot
    plt.figure(figsize=(8, 6.5), dpi=500)
    plt.scatter(x1, y1, marker="x", c="b", s=30, linewidth=0.75)
    plt.scatter(
        x2, y2, marker="o", edgecolors="blue", facecolors="none", s=30, linewidth=0.75
    )
    plt.plot(x, y, "b", linewidth=0.75)
    plt.contour(XX, YY, Z1, cmap="jet", linewidths=0.75)
    plt.contour(XX, YY, Z2, cmap="jet", linewidths=0.75)

    plt.xlim(-2, 7)
    plt.ylim(-7, 1)
    plt.tick_params(axis="both", direction="in", top=True, right=True)

    plt.savefig(save_path, format="svg", bbox_inches="tight", pad_inches=0.05)


if __name__ == "__main__":
    plot_gda()
