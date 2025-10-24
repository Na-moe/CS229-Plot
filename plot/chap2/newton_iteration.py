from pathlib import Path

import numpy as np

from matplotlib import pyplot as plt

from plot.utils import config_mpl, get_save_path


def f(x):
    return 0.2886 * (x - 1.3) * (x**2 + 3.1956 * x + 9.5991)


def df(x):
    return 0.2886 * (3 * x**2 + 3.7912 * x + 5.44482)


def set_ax(ax):
    ax.tick_params(axis="both", direction="in", top=True, right=True)
    ax.set_xlim(1, 5)
    ax.set_xticks(np.arange(1, 5.1, 0.5))
    ax.set_ylim(-10, 60)
    ax.set_yticks(np.arange(-10, 61, 10))
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")


def plot_x(ax, x, y, line_x, line_y):
    ax.plot(x, y, color="blue", linewidth=1.0)
    ax.plot(line_x, line_y, color="blue", linewidth=1.0, linestyle="-")


def plot_newton_iteration():
    # Configure Matplotlib
    config_mpl()

    # Data Preparing
    x = np.linspace(1, 5, 100)
    fx = f(x)
    zeros = np.zeros_like(x)

    f45 = f(4.5)
    x45 = np.linspace(-10, f45, 100)
    full45 = np.full_like(x45, 4.5)
    tan45 = df(4.5) * (x - 4.5) + f(4.5)

    f28 = f(2.8)
    x28 = np.linspace(-10, f28, 100)
    full28 = np.full_like(x28, 2.8)
    tan28 = df(2.8) * (x - 2.8) + f(2.8)

    # Get Save Path
    save_path = get_save_path(Path(__file__))

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(27, 6), dpi=500)

    _ = [set_ax(ax) for ax in axes]
    _ = [plot_x(ax, x, fx, x, zeros) for ax in axes]
    _ = [plot_x(ax, x, tan45, full45, x45) for ax in axes[1:]]
    _ = [plot_x(ax, x, tan28, full28, x28) for ax in axes[2:]]

    fig.savefig(save_path, format="svg", bbox_inches="tight", pad_inches=0.05)


if __name__ == "__main__":
    plot_newton_iteration()
