from .gaussian_scale import plot_gaussian_scale
from .gaussian_cov import plot_gaussian_cov
from .gaussian_shift import plot_gaussian_shift
from .gaussian_contour1 import plot_gaussian_contour1
from .gaussian_contour2 import plot_gaussian_contour2
from .gda import plot_gda


def plot_all():
    plot_gaussian_scale()
    plot_gaussian_cov()
    plot_gaussian_shift()
    plot_gaussian_contour1()
    plot_gaussian_contour2()
    plot_gda()


if __name__ == "__main__":
    plot_all()
