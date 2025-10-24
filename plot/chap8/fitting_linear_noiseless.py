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
    ax.set_title("在无噪声数据集上拟合线性模型")
    ax.legend(
        [f"{data_type}数据", "真实 h*", "最佳线性拟合模型"],
        loc="upper left",
        fontsize=12,
    )


def plot_fitting_linear_noiseless():
    # Configure Matplotlib
    config_mpl()

    # Data Loading
    train = pd.read_csv("data/chap8/train.csv", dtype=float)
    test = pd.read_csv("data/chap8/test.csv", dtype=float)

    XX = np.concatenate((train["x"], test["x"]))
    YY = np.concatenate((train["y"], test["y"]))
    p = Polynomial.fit(XX, YY, 2)

    Xn = np.linspace(0.05, 0.95, 20)
    Yn = p(Xn)

    p1n = Polynomial.fit(Xn, Yn, 1)

    xx = np.linspace(0, 1, 100)

    # Get Save Path
    save_path = get_save_path(Path(__file__))

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=500)
    ax.scatter(Xn, Yn, marker="x", color="blue")
    ax.plot(xx, p(xx), color="black", linewidth=1.0)
    ax.plot(xx, p1n(xx), color="red", linewidth=1.0, linestyle="-")
    set_ax(ax, "训练")

    fig.savefig(save_path, format="svg", bbox_inches="tight", pad_inches=0.05)


if __name__ == "__main__":
    plot_fitting_linear_noiseless()
