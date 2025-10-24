from pathlib import Path

from numpy.polynomial.polynomial import Polynomial
import pandas as pd

from matplotlib import pyplot as plt

from plot.utils import config_mpl, get_save_path


def set_ax(ax):
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 4.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.tick_params(axis="both", direction="in", top=True, right=True)


def plot_regression(ax, x, y, X, Y):
    set_ax(ax)
    ax.plot(x, y, "x", color="blue", markersize=3.5, markeredgewidth=0.4)
    ax.plot(X, Y, color="blue", linewidth=1.0)


def plot_polynomial_regression():
    # Configure Matplotlib
    config_mpl()

    # Data Loading
    data = pd.read_csv("data/chap1/polynomial-data.csv", dtype=float)
    x = data["x"]
    y = data["y"]

    linear_regression = Polynomial.fit(x, y, 1)
    quadratic_regression = Polynomial.fit(x, y, 2)
    quintic_regression = Polynomial.fit(x, y, 5)
    regressions = [
        linear_regression,
        quadratic_regression,
        quintic_regression,
    ]

    # Get Save Path
    save_path = get_save_path(Path(__file__))

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(27, 6), dpi=500)
    for ax, regression in zip(axes, regressions):
        plot_regression(ax, x, y, *regression.linspace(domain=[0, 7]))

    fig.savefig(save_path, format="svg", bbox_inches="tight", pad_inches=0.05)


if __name__ == "__main__":
    plot_polynomial_regression()
