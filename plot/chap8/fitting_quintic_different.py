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
    ax.legend(
        [f"{data_type}数据", "最佳五次拟合模型"],
        loc="upper center",
        fontsize=12,
    )


def plot_fitting_quintic_different():
    # Configure Matplotlib
    config_mpl()

    # Data Loading
    data = pd.read_csv("data/chap8/samples.csv", dtype=float)
    x1, y1, x2, y2, x3, y3 = (
        data["x1"],
        data["y1"],
        data["x2"],
        data["y2"],
        data["x3"],
        data["y3"],
    )

    p5_1 = Polynomial.fit(x1, y1, 5)
    p5_2 = Polynomial.fit(x2, y2, 5)
    p5_3 = Polynomial.fit(x3, y3, 5)

    xx = np.linspace(0, 1, 100)

    # Get Save Path
    save_path = get_save_path(Path(__file__))

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 4), dpi=500)
    _ = [
        ax.scatter(x, y, marker="x", color="blue")
        for ax, x, y in zip(axes, (x1, x2, x3), (y1, y2, y3))
    ]
    _ = [
        ax.plot(xx, p5(xx), color="red", linewidth=1.0, linestyle="-")
        for ax, p5 in zip(axes, (p5_1, p5_2, p5_3))
    ]
    _ = [set_ax(ax, "训练") for ax in axes]

    # set whole title
    axes[1].set_title("在不同数据集上拟合五次模型", pad=16.0, fontsize=16)

    fig.savefig(save_path, format="svg", bbox_inches="tight", pad_inches=0.05)


if __name__ == "__main__":
    plot_fitting_quintic_different()
