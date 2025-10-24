from .fitting_dataset import plot_fitting_dataset
from .fitting_linear import plot_fitting_linear
from .fitting_linear_large import plot_fitting_linear_large
from .fitting_linear_noiseless import plot_fitting_linear_noiseless
from .fitting_quadratic import plot_fitting_quadratic
from .fitting_quintic import plot_fitting_quintic
from .fitting_quintic_large import plot_fitting_quintic_large
from .fitting_quintic_different import plot_fitting_quintic_different


def plot_all():
    plot_fitting_dataset()
    plot_fitting_linear()
    plot_fitting_linear_large()
    plot_fitting_linear_noiseless()
    plot_fitting_quadratic()
    plot_fitting_quintic()
    plot_fitting_quintic_large()
    plot_fitting_quintic_different()


if __name__ == "__main__":
    plot_all()
