from pathlib import Path

import numpy as np
from numpy.polynomial.polynomial import Polynomial
import pandas as pd

from matplotlib import pyplot as plt

from plot.utils import config_mpl, get_save_path


def set_ax(ax, data_type: str):
    ax.set_xlim(0, 1.0)
    ax.set_xticks(np.arange(0, 1.1, 0.2))
    ax.set_ylim(-0.4, 1.5)
    ax.set_yticks(np.arange(0, 1.6, 0.5))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("在大规模数据集上拟合五次模型")
    ax.legend(
        [f"{data_type}数据", "真实 h*", "最佳五次拟合模型"],
        loc="upper left",
        fontsize=12,
    )


def plot_fitting_quintic_large():
    # Configure Matplotlib
    config_mpl()

    # Data Loading
    train = pd.read_csv("data/chap8/train.csv", dtype=float)
    test = pd.read_csv("data/chap8/test.csv", dtype=float)

    XX = np.concatenate((train["x"], test["x"]))
    YY = np.concatenate((train["y"], test["y"]))
    p = Polynomial.fit(XX, YY, 2)

    xx = np.linspace(0, 1, 100)

    seed = 42
    np.random.seed(seed)
    noise = np.random.rand(*xx.shape)
    X1 = np.concatenate((train["x"], xx))
    Y1 = np.concatenate((train["y"], p(xx) + 0.5 * noise - 0.25))
    p5 = Polynomial.fit(X1, Y1, 5)

    # Get Save Path
    save_path = get_save_path(Path(__file__))

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=500)
    ax.scatter(X1, Y1, marker="x", color="blue")
    ax.plot(xx, p(xx), color="black", linewidth=1.0)
    ax.plot(xx, p5(xx), color="red", linewidth=1.0, linestyle="-")
    set_ax(ax, "训练")

    fig.savefig(save_path, format="svg", bbox_inches="tight", pad_inches=0.05)


if __name__ == "__main__":
    plot_fitting_quintic_large()
