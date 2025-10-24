from pathlib import Path

import numpy as np

from matplotlib import pyplot as plt

from plot.utils import config_mpl, get_save_path


def plot_sigmoid():
    # Configure Matplotlib
    config_mpl()

    # Data Preparing
    x = np.linspace(-5, 5, 100)
    sigmoid = 1 / (1 + np.exp(-x))

    # Get Save Path
    save_path = get_save_path(Path(__file__))

    # Plot
    plt.figure(figsize=(8, 6), dpi=500)
    plt.plot(x, sigmoid, color="blue", linewidth=1.0)

    plt.xlim(-5, 5)
    plt.xticks(np.arange(-5, 6, 1))
    plt.ylim(0, 1)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.tick_params(axis="both", direction="in", top=True, right=True)
    plt.xlabel("z")
    plt.ylabel("g(z)")

    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.05)


if __name__ == "__main__":
    plot_sigmoid()
