from pathlib import Path

import numpy as np

from matplotlib import pyplot as plt

from plot.utils import config_mpl, get_save_path


def plot_activations():
    # Configure Matplotlib
    config_mpl()

    # Data Preparing
    x = np.linspace(-4, 4, 200)
    relu = np.maximum(x, 0)
    sigmoid = 1 / (1 + np.exp(-x))
    tanh = np.tanh(x)
    leaky_relu = np.maximum(x, 0.3 * x)
    gelu = 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    softplus = np.log(1 + np.exp(x))

    # Get Save Path
    save_path = get_save_path(Path(__file__))

    # Plot
    plt.figure(figsize=(9, 5), dpi=600)
    plt.plot(x, relu, "-", color="blue", linewidth=1.0, label="ReLU")
    plt.plot(x, sigmoid, "-", color="orange", linewidth=1.0, label="sigmoid")
    plt.plot(x, tanh, "-", color="green", linewidth=1.0, label="tanh")
    plt.plot(x, leaky_relu, "-", color="red", linewidth=1.0, label=r"leaky ReLU, γ=0.3")
    plt.plot(x, gelu, "-", color="purple", linewidth=1.0, label="GELU")
    plt.plot(x, softplus, "-", color="brown", linewidth=1.0, label=r"Softplus, β=1")

    plt.xlim(-4.3, 4.3)
    plt.xticks(np.arange(-4, 5, 1))
    plt.ylim(-1.3, 4.2)
    plt.yticks(np.arange(-1, 5, 1))
    plt.xlabel("z")
    plt.ylabel(r"$\sigma$(z)")
    plt.legend(loc="upper left", fontsize=10)

    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.05)


if __name__ == "__main__":
    plot_activations()
