from .house_dataset import plot_house_dataset
from .house_dataset_linear_regression import plot_house_dataset_linear_regression
from .polynomial_regression import plot_polynomial_regression
from .gradient_descent_trajectory import plot_gradient_descent_trajectory


def plot_all():
    plot_house_dataset()
    plot_house_dataset_linear_regression()
    plot_polynomial_regression()
    plot_gradient_descent_trajectory()


if __name__ == "__main__":
    plot_all()
