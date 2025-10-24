from .sigmoid import plot_sigmoid
from .newton_iteration import plot_newton_iteration


def plot_all():
    plot_sigmoid()
    plot_newton_iteration()


if __name__ == "__main__":
    plot_all()
