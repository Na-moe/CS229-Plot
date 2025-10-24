from pathlib import Path

import numpy as np
from numpy.polynomial.polynomial import Polynomial
import pandas as pd

from matplotlib import pyplot as plt

from plot.utils import config_mpl, get_save_path


def plot_house_dataset_linear_regression():
    # Configure Matplotlib
    config_mpl()

    # Data Loading
    data = pd.read_csv("data/chap1/portland-houses.csv", dtype=float)
    area: pd.Series[float] = data["area"]
    price: pd.Series[float] = data["price"] / 1000

    linear_regression = Polynomial.fit(area, price, 1)
    X = np.linspace(500, 5000, 100)
    Y = linear_regression(X)

    # Get Save Path
    save_path = get_save_path(Path(__file__))

    # Plot
    plt.figure(figsize=(8, 6), dpi=500)
    plt.plot(area, price, "x", color="blue", markersize=3.5, markeredgewidth=0.4)
    plt.plot(X, Y, color="blue", linewidth=1.0)
    plt.tick_params(axis="both", direction="in", top=True, right=True)
    plt.xlim(500, 5000)
    plt.xticks(np.arange(500, 5100, 500))
    plt.ylim(-30, 1030)
    plt.yticks(np.arange(0, 1100, 100))
    plt.xlabel("平方英尺")
    plt.ylabel("价格（千美元）")
    plt.title("房价")
    plt.savefig(save_path, format="svg", bbox_inches="tight", pad_inches=0.05)


if __name__ == "__main__":
    plot_house_dataset_linear_regression()
