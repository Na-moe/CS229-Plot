from pathlib import Path

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from plot.utils import config_mpl, get_save_path


def plot_dataset_discrete():
    # Configure Matplotlib
    config_mpl()

    # Data Loading
    data = pd.read_csv("data/chap15/data.csv", dtype=float)
    X, Y = data["x"], data["y"]

    YY = []

    for i in range(1, 8, 1):
        y = Y[(X <= i + 1) & (X >= i)].to_numpy()
        y_min, y_max = y.min(), y.max()
        yy = np.linspace(y_min, y_max, 100)[:, None]
        y_star = yy[np.square(yy - y[None, :]).sum(axis=1).argmin()].squeeze()
        YY.append(np.full(100, y_star, dtype=float))

    YY = np.array(YY).flatten()
    XX = np.linspace(1, 8, 700)

    # Get Save Path
    save_path = get_save_path(Path(__file__))

    # Plot
    plt.figure(figsize=(8, 6), dpi=500)
    plt.plot(X, Y, "x", color="blue", markersize=3.5, markeredgewidth=0.4)
    plt.plot(XX, YY, "-", color="blue", markersize=3.5, markeredgewidth=0.4)

    plt.tick_params(axis="both", direction="in", top=True, right=True)
    plt.xlim(1, 8)
    plt.xticks(np.arange(1, 9, 1))
    plt.ylim(1.5, 5.5)
    plt.yticks(np.arange(1.5, 5.6, 0.5))
    plt.xlabel("x")
    plt.ylabel("y")

    plt.savefig(save_path, format="svg", bbox_inches="tight", pad_inches=0.05)


if __name__ == "__main__":
    plot_dataset_discrete()
